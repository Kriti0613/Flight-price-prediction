import html
import json
import os
from pathlib import Path
from urllib.parse import parse_qs

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "flight_price_model.pkl"))
TRAINING_DATA_PATH = Path(
    os.getenv("TRAINING_DATA_PATH", BASE_DIR / "data" / "train.csv")
)
CATEGORICAL_FEATURES = [
    "airline",
    "flight",
    "source",
    "departure",
    "stops",
    "arrival",
    "destination",
    "class",
]
NUMERIC_FEATURES = ["duration", "days_left"]
USER_FEATURES = ["airline", "class", "source", "destination"]


def load_training_metadata():
    training_data = pd.read_csv(TRAINING_DATA_PATH)

    category_options = {}
    category_defaults = {}
    for column in CATEGORICAL_FEATURES:
        values = training_data[column].dropna().astype(str)
        if values.empty:
            raise ValueError(f"Training data has no values for {column!r}.")
        category_options[column] = sorted(values.unique().tolist())
        category_defaults[column] = values.mode().iloc[0]
        training_data[column] = training_data[column].fillna(category_defaults[column])

    if "price" in training_data.columns:
        first_quartile = training_data["price"].quantile(0.25)
        third_quartile = training_data["price"].quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        training_data = training_data.loc[
            training_data["price"].between(lower_bound, upper_bound)
        ]

    numeric_defaults = training_data[NUMERIC_FEATURES].median().to_dict()
    numeric_data = training_data[NUMERIC_FEATURES].fillna(numeric_defaults)
    scaler = MinMaxScaler().fit(numeric_data)

    return category_options, category_defaults, numeric_defaults, scaler


model = joblib.load(MODEL_PATH)
feature_columns = list(model.feature_names_in_)
(
    category_options,
    category_defaults,
    numeric_defaults,
    feature_scaler,
) = load_training_metadata()

app = FastAPI(
    title="Flight Price Predictor",
    description="Predict a flight price using airline, class, source, and destination.",
)


def render_page(predicted_price=None, error_message=None, submitted_values=None):
    submitted_values = submitted_values or {}
    controls = []
    for column in USER_FEATURES:
        label = column.title()
        options = []
        for value in category_options[column]:
            selected = " selected" if value == submitted_values.get(column) else ""
            safe_value = html.escape(value, quote=True)
            options.append(
                f'<option value="{safe_value}"{selected}>{html.escape(value)}</option>'
            )
        controls.append(
            f'<label for="{column}">{label}</label>'
            f'<select id="{column}" name="{column}" required>'
            f'{"".join(options)}</select>'
        )

    result = ""
    if predicted_price is not None:
        result = (
            '<p class="result">Estimated flight price: '
            f'<strong>{predicted_price:,.2f}</strong></p>'
        )
    elif error_message:
        result = f'<p class="error">{html.escape(error_message)}</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flight Price Predictor</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: #f3f6f4; color: #1d2925; }}
    main {{ max-width: 620px; margin: 8vh auto; padding: 32px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    p {{ color: #52625b; }}
    form {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 28px; }}
    label {{ display: grid; gap: 7px; font-weight: 650; }}
    select {{ width: 100%; min-height: 44px; padding: 8px 10px; border: 1px solid #bbc9c1; border-radius: 4px; background: white; color: inherit; }}
    button {{ grid-column: 1 / -1; min-height: 46px; border: 0; border-radius: 4px; background: #12664f; color: white; font: inherit; font-weight: 700; cursor: pointer; }}
    .result, .error {{ grid-column: 1 / -1; margin: 4px 0 0; padding: 14px; background: #e2eee8; color: #164a3b; }}
    .error {{ background: #fbe8e4; color: #822f24; }}
    @media (max-width: 520px) {{ main {{ margin: 0 auto; padding: 24px 18px; }} form {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <h1>Flight Price Predictor</h1>
    <p>Choose the flight details to estimate its price.</p>
    <form action="/predict" method="post">
      {"".join(controls)}
      <button type="submit">Predict price</button>
      {result}
    </form>
  </main>
</body>
</html>"""


def predict_price(payload):
    if not isinstance(payload, dict):
        raise ValueError("Submit a JSON object or an HTML form.")

    row = dict(category_defaults)
    row.update(numeric_defaults)
    for column in USER_FEATURES:
        entered_value = str(payload.get(column, "")).strip()
        category_lookup = {value.casefold(): value for value in category_options[column]}
        canonical_value = category_lookup.get(entered_value.casefold())
        if canonical_value is None:
            choices = ", ".join(category_options[column])
            raise ValueError(f"Invalid {column}. Choose one of: {choices}")
        row[column] = canonical_value

    encoded = pd.get_dummies(
        pd.DataFrame([row]), columns=CATEGORICAL_FEATURES
    ).reindex(columns=feature_columns, fill_value=0)
    encoded[NUMERIC_FEATURES] = feature_scaler.transform(encoded[NUMERIC_FEATURES])
    return float(model.predict(encoded)[0])


@app.get("/", response_class=HTMLResponse)
async def home():
    return render_page()


@app.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": USER_FEATURES,
                        "properties": {
                            column: {
                                "type": "string",
                                "enum": category_options[column],
                            }
                            for column in USER_FEATURES
                        },
                    }
                },
                "application/x-www-form-urlencoded": {
                    "schema": {
                        "type": "object",
                        "required": USER_FEATURES,
                        "properties": {
                            column: {"type": "string"} for column in USER_FEATURES
                        },
                    }
                },
            },
        }
    },
)
async def predict(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    is_form = "application/x-www-form-urlencoded" in content_type

    try:
        if "application/json" in content_type:
            payload = await request.json()
        elif is_form:
            parsed_form = parse_qs((await request.body()).decode("utf-8"))
            payload = {key: values[-1] for key, values in parsed_form.items()}
        else:
            raise ValueError("Use JSON or submit the HTML form.")

        predicted_price = predict_price(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if is_form:
            return HTMLResponse(
                render_page(error_message=str(error), submitted_values=locals().get("payload")),
                status_code=422,
            )
        return JSONResponse(status_code=422, content={"detail": str(error)})

    if is_form:
        return HTMLResponse(
            render_page(predicted_price=predicted_price, submitted_values=payload)
        )
    return {"predicted_price": predicted_price, "currency": "INR"}