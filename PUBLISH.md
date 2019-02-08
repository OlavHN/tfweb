## Reminder

- Make sure ~/.pypirc is setup with pypi url: repository = https://upload.pypi.org/legacy/
- Change version number in `setup.py`
- `git tag v{VERSION} master && git push --tags`
- `python setup.py sdist`
- `twine upload dist/*`
