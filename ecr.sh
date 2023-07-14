REGION=eu-central-1
ACCOUNT_ID=643202173500
REPO_NAME=mlops_cop
VERSION=0.0.1
IMAGETAG="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${VERSION}"

docker build . -t $IMAGETAG
docker push $IMAGETAG