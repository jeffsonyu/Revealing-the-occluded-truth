# rfhand-serial-broker

VR Interface for the glove using RFUniverse

## Get Started


```shell
git clone https://github.com/mvig-robotflow/rfhand-serial-broker
cd rfhand-serial-broker
```


Install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Supposing that the name `glove` is used for conda environment:

```shell
conda create -n glove python=3.9
conda activate glove
```

Install the dependencies

```shell
pip install -r requirements.txt
```

## Work with RFUniverse

First we run `manifests/MannoHand_Windows`, it will create a `config.json` under the directory `.rfuniverse` in your root directory. Please make sure the content of the file is valid:

```json
{
  "$type": "RFUniverse.RFUniverseMain+ConfigData, Assembly-CSharp",
  "assets_path": "",
  "executable_file": "C:\\Users\\username\\Desktop\\rfhand-serial-broker\\manifests\\ManoHand_Windows\\ManoHand.exe"
}
```

And we can run

```python
python tests/test_serial_broker_with_rfu.py
```

to start the RFUniverse Interface, use WASD to move, and use the mouse wheel to zoom in and out

## Tasks

### task_generate_pseudo_data

Generate the test data

```shell
python tasks/task_generate_pseudo_data.py --output=./tests/pseudo_data.txt
```

## Tests

### test_parser_sync

Reading fake input from a file

### test_serial_broker

Reads test data from the serial port.

Change the content of the file `manifests/config.yaml` to set the serial port and baud rate

```shell
python tests/test_serial_broker.py --config=./manifests/config.yaml --plot
```

It will print the parsed Numpy array and plot the image
