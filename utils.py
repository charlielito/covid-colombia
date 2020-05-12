import os
from io import BytesIO
from pathlib import Path

import boto3
import botocore
import geopy
import numpy as np
import pandas as pd
import unidecode
from geopy.geocoders import Nominatim
from PIL import Image
from sodapy import Socrata
from pathlib import Path


S3 = boto3.client("s3")
boto_resource = boto3.resource("s3")
geolocator = Nominatim(user_agent="test")
api_key = os.getenv("DATA_KEY")


MONTHS_LOOKUP = [
    None,
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]


def get_data(database_id, location_file, use_cache):
    client = Socrata("www.datos.gov.co", api_key)
    results = client.get(database_id, limit=100000)

    df = pd.DataFrame.from_records(results)
    df["location"] = df.apply(
        lambda row: f"{row.ciudad_de_ubicaci_n} {row.departamento}", axis=1
    )
    # remove accents
    df["location"] = df.location.apply(lambda x: str(unidecode.unidecode(x)).upper())
    cities = pd.unique(df.location)

    print("Checking location file...")
    if use_cache and file_exists(location_file):
        print("File exists now reading")
        location_df = pd.read_csv(location_file)
        print("Done")

    else:
        print("Could not locate file, calculating from scratch")
        maybe_mkdirs(location_file)

        print("Calculating...")
        location_df = get_location_df(cities)
        print("Ready! Writing to remote...")
        location_df.to_csv(location_file, index=False)
        print("Done")

    print("Check if cities matches")
    if len(location_df.location.values) != len(cities):
        new_cities = set(cities) - set(location_df.location.values)
        print("Difference", new_cities)
        print("Calculating new cities")
        _location_df = get_location_df(list(new_cities))
        location_df = location_df.append(_location_df)
        print("Done!")

        # save updated version
        print("Ready! Writing to remote...")
        location_df.to_csv(location_file, index=False)
        print("Done")

    return df, location_df


def bytes2image(image_bytes):
    with BytesIO(image_bytes) as image_bytes:
        image = Image.open(image_bytes)
        image = np.array(image, copy=True)
    return image


def get_lat_lon(city):
    # print(city)
    geocode_obj = geolocator.geocode(city)
    return (
        (geocode_obj.latitude, geocode_obj.longitude)
        if geocode_obj is not None
        else (None, None)
    )


def get_location_df(cities, retries=10):
    location_df = pd.DataFrame.from_dict(dict(location=cities))

    for _ in range(retries):
        try:
            location_df["lat_lon"] = location_df.location.apply(
                lambda city: get_lat_lon(city)
            )
            break
        except geopy.exc.GeocoderTimedOut:
            print("Service time out, retrying...")

    # Add santa marta because it fails to get the gps values
    location_df[location_df.location == "SANTA MARTA SANTA MARTA D.T. Y C"][
        "lat_lon"
    ] = (
        11.2422289,
        -74.2055606,
        0,
    )

    location_df["lat"] = location_df.lat_lon.apply(lambda t: t[0])
    location_df["lon"] = location_df.lat_lon.apply(lambda t: t[1])
    location_df.drop(columns=["lat_lon"], inplace=True)
    return location_df


def get_groupby_location(df, location_df):
    gb = pd.DataFrame({"cases": df.groupby("location").size()}).reset_index()
    gb["lat"] = gb.location.apply(
        lambda loc: location_df[location_df.location == loc].iloc[0].lat
    )
    gb["lon"] = gb.location.apply(
        lambda loc: location_df[location_df.location == loc].iloc[0].lon
    )
    return gb


def get_platform(file_path):
    split = file_path.split("://")
    if len(split) == 1:
        return "local"
    return split[0]


def write_file(file_path, content):
    platform = get_platform(file_path)
    if platform == "local":
        with open(file_path, mode="w") as f:
            f.write(content)
    elif platform == "s3":
        write_s3_file(file_path, content)
    else:
        raise NotImplementedError


def file_exists(file_path):
    platform = get_platform(file_path)

    if platform == "local":
        return os.path.exists(file_path)
    elif platform == "s3":
        file_path = file_path.split("/")
        bucket = file_path[2]
        key = "/".join(file_path[3:])
        try:
            boto_resource.Object(bucket, key).load()
            return True
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                # Something else has gone wrong.
                raise
    else:
        raise NotImplementedError


def maybe_mkdirs(file_path):
    platform = get_platform(file_path)
    if platform == "local":
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def write_s3_file(file_path, content):
    file_path = file_path.split("/")
    bucket = file_path[2]
    key = "/".join(file_path[3:])

    S3.put_object(Body=content, Bucket=bucket, Key=key, ContentType="text/html")
