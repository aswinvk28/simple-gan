echo "\nDownloading model..\n"

BASE_PATH="https://models-openvino.s3.eu-west-2.amazonaws.com"
OPENVINO_PATH=$1

path_openvino=$(which $OPENVINO_PATH 2> /dev/null)

if [[test -z "$path_openvino"]]
then
      echo "OpenVINO path is not available..\n"
fi

while getopts "p:r:" opt; do
  case $opt in
    p)  
        PRECISION=$OPTARG;;
    r)  
        RUN_PROJECT=1;;
    *) 
        echo "Invalid option: -$OPTARG" >&2;;  
  esac
done

if test -z "$(ls downloaded_models)"
then
    mkdir -p ./downloaded_models/FP16
    mkdir -p ./downloaded_models/FP32
else
    echo "Directory exists..\n"
fi

if test -n "$PRECISION"
then
    wget -o "./downloaded_models/$PRECISION/mnist_generator.xml" "$BASE_PATH/$PRECISION/mnist_generator.xml"
    wget -o "./downloaded_models/$PRECISION/mnist_generator.bin" "$BASE_PATH/$PRECISION/mnist_generator.bin"
fi

echo "Downloaded Models and Weights..\n"

path27=$(which python2.7 2> /dev/null)
path3=$(which python3 2> /dev/null)

if [[test -n "$path3"]] && [[test -n "$RUN_PROJECT"]]
then
      echo "python3 available.\n"
      echo "Installing python3 packages..\n"
      python3 -m pip install -r requirements.txt
fi

if [[test -n "$RUN_PROJECT"]]
then
    source "$OPENVINO_PATH/bin/setupvars.sh" -pyver "$VERSION"
    python3 $OPENVINO_PATH
fi
