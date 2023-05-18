# Release Checklist #

- run tests and record results wth version number and time
- bump version number in celestine/__init__.py
- bump version number in sphinx documentation in docs/source/conf.py
- create docs by running make latexpdf in docs/ directory
- update Changelog.md
- tag commit with version number
