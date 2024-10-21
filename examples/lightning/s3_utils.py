import boto3

def parse_s3_uri(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI")
    
    # Remove the "s3://" prefix
    s3_uri = s3_uri[5:]
    
    # Split the remaining part into bucket and prefix
    parts = s3_uri.split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    return bucket_name, prefix

def list_s3_files(s3_uri):
    bucket_name, prefix = parse_s3_uri(s3_uri)
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    paths = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                paths.append(f"s3://{bucket_name}/{obj['Key']}")

    return sorted(paths)
