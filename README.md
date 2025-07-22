# ssd

Re-implements the original SSD model using PyTorch. The main focus of this work is to make it easier to:

1. Re-train the SSD model.
2. Productionise SSD networks - with OpenVINO or TensorRT.

## Install

### Development

#### Python virtual environment

This project uses [PDM](https://pdm-project.org/en/latest/) to manage the Python environment. To install this run:

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

To install the project dependencies (including dev dependencies) run:

```bash
pdm install -d
```

#### Linting & static analysis

Linting and static analysis is managed by [Trunk](https://docs.trunk.io/). To install this run:

```bash
curl https://get.trunk.io -fsSL | bash
```

This will automatically check for issues on `commit` and `push`. If you wish to run the analysis manually you can:

```bash
trunk check
```

#### Testing

`pytest` is used for testing. To run the tests first ensure the Python virtual environment is enabled, then run:

```bash
pytest tests/
```

#### IDE

This repo was developed using VSCode. For an improved user experience it is recommended to use VSCode and install the `Python` extension - this will allow you to debug the code and run tests from the editor.
