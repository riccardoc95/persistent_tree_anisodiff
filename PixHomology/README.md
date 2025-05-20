![Alt text](https://github.com/riccardoc95/PixHomology/blob/main/docs/logo_color.png)

# Calculation of Persistent Homology for Pixel Data
[![version](https://img.shields.io/badge/version-0.0.1alpha-yellow.svg)](https://github.com/riccardoc95/PixHomology/releases/tag/alpha_v0.0.1)
[![version](https://img.shields.io/badge/python-11.5-version.svg)](https://www.python.org/)
[![version](https://img.shields.io/badge/gcc-15.0.0-version.svg)](https://gcc.gnu.org/)

PixHomology is an open-source software for image processing and analysis focused on persistent homology computation. It provides a set of tools and algorithms to explore the topological features of 2D images, enabling users to extract meaningful information about the underlying structures. 

## Features

- In its initial release, the software enables the computation of 0-dimensional homology on a 2D image.
- The construction of simplicial complexes involves establishing connections between each pixel and its 8 neighboring pixels.
  
## Installation

**Optional** (create new environment):
```bash
conda create -n pixh python=3.11
conda activate pixh
```

### Python package
```bash
SOON
```

### Building
> To build the source code, ensure you have CMake installed on your system. You can download CMake from [cmake.org](cmake.org) or install it using your package manager. 
To install PixHomology, follow these steps:

```bash
git clone https://github.com/riccardoc95/PixHomology.git
cd PixHomology
pip install .
```

For detailed installation instructions, please refer to the [Installation Guide](docs/installation.md).

## Usage
- To test the package:
```bash
cd tests
python test.py
```

- Here is a basic example of using PixHomology in Python:

```python
import pixhomology as px
image = np.random.rand(10,10)
dgm = px.computePH(image)
...
```
For more examples and detailed usage instructions, check out the [Usage Guide](docs/usage.md).

- To test the performance of PixHomology:
> Please note that the installation of other packages and datasets download are required
 ```bash
cd tests/ripser_comparison
pip install -r requirements.txt
python test.py
```

## Documentation

Comprehensive documentation is available in the [Docs](docs/) directory. This includes installation guides, usage examples, and API references.

## Contributing

We welcome contributions from the community. If you want to contribute to PixHomology, please check the [Contribution Guidelines](CONTRIBUTING.md) for more details.

## License

PixHomology is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

---
