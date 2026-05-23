# Build docker image
docker build -t image .
# Run docker image with files
sudo docker run -it \
-v $(pwd)/docker_test/model:/physionet/model \
-v $(pwd)/docker_test/test_data:/physionet/test_data \
-v $(pwd)/docker_test/test_outputs:/physionet/test_outputs \
-v $(pwd)/docker_test/training_data:/physionet/training_data \
image bash training.sh
