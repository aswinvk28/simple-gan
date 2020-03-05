echo "\nDownloading model..\n"

BASE_PATH="https://models-openvino.s3.eu-west-2.amazonaws.com"
OPENVINO_PATH=$1
VERSION=$2

while getopts "p:r:o:v" opt; do
  case $opt in
    p)  
        PRECISION=$OPTARG;;
    r)  
        RUN_PROJECT=1;;
    o) 
        OPENVINO_PATH=$OPTARG;;
    v) 
        VERSION=$OPTARG;;
    *) 
        echo "Invalid option: -$OPTARG" >&2;;  
  esac
done

path_openvino=$(ls $OPENVINO_PATH 2> /dev/null)

if test -z "$path_openvino"
then
      echo "OpenVINO path is not available..\n"
fi

if test -z "$(ls downloaded_models 2> /dev/null)"
then
    mkdir -p ./downloaded_models/FP16
    mkdir -p ./downloaded_models/FP32
else
    echo "Directory exists..\n"
fi

if [[ -n $PRECISION ]] && [[ -z "$(ls downloaded_models 2> /dev/null)" ]]; then
    wget -O "./downloaded_models/$PRECISION/mnist_generator.xml" "$BASE_PATH/$PRECISION/mnist_generator.xml"
    wget -O "./downloaded_models/$PRECISION/mnist_generator.bin" "$BASE_PATH/$PRECISION/mnist_generator.bin"
fi

echo "Downloaded Models and Weights..\n"

path27=$(which python2.7 2> /dev/null)
path3=$(which python3 2> /dev/null)

if [[ -n "$path3" ]] && [[ -n "$RUN_PROJECT" ]]; then
      echo "python3 available.\n"
      echo "Installing python3 packages..\n"
      python3 -m pip install -r requirements.txt
fi

if [[ -n "$RUN_PROJECT" ]]; then
    source "$OPENVINO_PATH/bin/setupvars.sh" -pyver "$VERSION"
    python3 $OPENVINO_PATH
fi
