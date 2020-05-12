import os
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytz
import typer

import utils

mapbox_token = os.getenv("MAPBOX_TOKEN")


def main(
    database_id: str = "gt2j-8ykr",
    use_cache: bool = False,
    # location_file: str = "location.csv",
    location_file: str = "s3://www.charlielito.ml/data/location.csv",
    viz: bool = False,
    output_html: str = "s3://www.charlielito.ml/index.html",
):
    df, location_df = utils.get_data(database_id, location_file, use_cache)

    gb = utils.get_groupby_location(df, location_df)
    center = location_df[location_df.location == "BOGOTA BOGOTA D.C."].iloc[0]

    fn = lambda x: np.log(x / 6 + 1)
    max_cases = fn(max(gb.cases))
    total = sum(gb.cases)
    print(f"TOTAL CASES: {total}",)

    print("Getting plotly figure...")
    utc_today = pytz.utc.localize(datetime.utcnow())
    col_today = utc_today.astimezone(pytz.timezone("America/Bogota"))
    today_str = col_today.strftime("%Y-%m-%d %I:%M %p")

    fig = go.Figure(
        data=[
            go.Scattermapbox(
                lat=gb["lat"],
                lon=gb["lon"],
                marker=go.scattermapbox.Marker(
                    size=40 * fn(gb.cases) / max_cases, color="red"
                ),
                text=[
                    f"{row.location} -> {row.cases} casos" for i, row in gb.iterrows()
                ],
                hoverinfo="text",
            ),
        ],
        layout=dict(
            title={
                "text": f"Casos COVID-19 en Colombia: {total} total <br>"
                f"Actualizado: {today_str}",
                "xanchor": "center",
                "x": 0.5,
                # "yanchor": "top",
                # "y": 0.95,
            },
            font=dict(family="Courier New, monospace", size=24, color="#7f7f7f"),
            mapbox=dict(
                style="dark",
                accesstoken=mapbox_token,
                zoom=5,
                center=go.layout.mapbox.Center(lat=center.lat, lon=center.lon),
            ),
            showlegend=False,
            hoverlabel=dict(font=dict(size=25)),
        ),
    )
    if viz:
        fig.show()

    print("Done!!")
    with StringIO() as str_buf:
        fig.write_html(str_buf)
        out_str = str_buf.getvalue()

    print("Writing html to s3")
    utils.write_file(output_html, out_str)
    print("Done")


if __name__ == "__main__":
    typer.run(main)
