import pydevd
from pydevd.pydevd_constants import IS_PYCHARM

# Create a list of configuration names to execute
config_names = ['auto_autopep8-dev_code', 'auto_autopep8-old_code']

# Loop through each configuration name and execute it
for config_name in config_names:
    # Create a debug launch info object
    launch_info = pydevd.create_launch_info_from_pydevd_params(
        launch_file=None,
        launch_module=None,
        launch_name=config_name,
        port=None,
        start_client=False,
        is_module=False,
    )

    # Launch the configuration
    pydevd.settrace(
        suspend=False,
        trace_only_current_thread=False,
        patch_multiprocessing=False,
        stop_at_frame=None,
        overwrite_prev_trace=True,
    )
    pydevd.exec_python_file(launch_info)
