To reset your Python environment in Jupyter Notebook, you can run the following commands using the `!` operator to execute shell commands:

### For `pip`:

```python
!pip freeze | xargs pip uninstall -y
!pip cache purge
```

### For `conda`:

```python
!conda list --export | cut -d '=' -f 1 | xargs conda remove -y
!conda clean --all --yes
```

Make sure to restart your Jupyter Notebook kernel after running these commands to apply the changes.