# Contributing to ROCm Kernel Playground

Thank you for your interest in contributing!

## Development Setup

### Prerequisites
- ROCm 5.0+ installed
- HIP compiler (hipcc)
- AMD GPU

```bash
git clone https://github.com/sudheerdevu/ROCm-Kernel-Playground.git
cd ROCm-Kernel-Playground
```

## Building Kernels

```bash
cd kernels/01_hello_hip
make
./hello_hip
```

## Adding New Tutorials

1. Create numbered directory: `kernels/0X_topic_name/`
2. Include:
   - `*.hip` source file
   - `Makefile`
   - `README.md` with explanation
3. Follow progressive difficulty

## Code Style

- Use HIP naming conventions
- Include comments explaining GPU concepts
- Add error checking with `HIP_CHECK`

## Pull Request Process

1. Fork repository
2. Create feature branch
3. Test on AMD GPU hardware
4. Submit PR

## License

Contributions are licensed under MIT License.
