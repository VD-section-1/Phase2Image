# Phase2Image
Project for Visual Data classes related to reconstruction of images from the phase spectrums.

## Preprocessing
To create a ready-to-use dataset, follow these steps:
1. Unzip the `dataset.zip` file into the `data/color` folder.
2. Execute the `preprocess_pictures.py` script.
This script will generate grayscale pictures and phase spectra.

## Guide
To ensure a smooth development experience, please adhere to the following rules while working on our Python project hosted on GitHub:
1. Follow the PEP8 style guide for consistent and readable code.
2. Adopt conventional commits to maintain a meaningful commit history.
3. Document your code using docstrings and a consistent documentation style. For example:
```python
"""
    Class that performs a specific task.

    :param file_name: String storing the name of the file.
    :param file_path: String storing the path to the file.
    :return: Float representing a result.
    :raise FileNotFoundError: Exception raised if the file doesn't exist.
"""
```
4. Implement your code as a module with a proper `__init__.py` file.
5. Test your code thoroughly.
6. Work only on your branch named "surname1_surname2".
7. Work in a folder named "surname1_surname2" created at the root of the project.
8. Conduct code reviews of code submitted by other members.
9. Utilize a .gitignore file to exclude unnecessary files.
10. Use Black for consistent code formatting.
11. Utilize a requirements.txt file to manage project dependencies. Include version specifications for dependencies to ensure reproducibility.
12. When the code is ready, create a pull request so it can be reviewed by others.
13. After obtaining three :thumbs_up: emojis under the pull request, the code will be merged with the master branch.
14. DO NOT push anything directly to the master branch.
15. Prepare documentation of the solution in a Markdown `Documentation.md` file for inclusion in the scientific paper later.
