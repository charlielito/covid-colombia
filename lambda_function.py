import json
from map import main


def lambda_handler(event, context):
    main(
        use_cache=True,
        location_file="s3://www.charlielito.ml/data/location.csv",
        output_html="s3://www.charlielito.ml/index.html",
    )
    return {"statusCode": 200, "body": json.dumps("Done with the work")}
