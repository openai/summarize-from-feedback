"""
Utilities related to launching and managing experiments.
"""

from functools import partial, wraps
from typing import List, get_type_hints

from summarize_from_feedback.utils import jobs
from summarize_from_feedback.utils.combos import bind, combos


def get_annotation_of_only_argument(fn):
    annotations = get_type_hints(fn).values()
    if len(annotations) != 1:
        raise ValueError(
            f"fn {fn} has {len(annotations)} arguments, but we wanted 1: {annotations}"
        )
    ty, = annotations
    return ty


def get_experiment_jobs(name, launch_fn, trials, hparam_class=None) -> List[jobs.Job]:
    if hparam_class is None:
        hparam_class = get_annotation_of_only_argument(launch_fn)

    # Maps experiment def key to argument name for jobs.launch; these get pulled out
    launch_kwarg_keys = dict(mpi="mpi", mode="mode")

    to_launch = []
    for trial in trials:
        descriptors = []
        trial_bindings = []
        trial_bindings_dict = {}  # only used for message

        # Extract bindings & descriptors from the trial
        for k, v, s in trial:
            if k is not None:
                if k in trial_bindings_dict:
                    print(f"NOTE: overriding key {k} from {trial_bindings_dict[k]} to {v}")
                trial_bindings.append((k, v))
                trial_bindings_dict[k] = v
            if "descriptor" in s and s["descriptor"] is not "":
                descriptors.append(str(s["descriptor"]))

        # Pull out arguments for jobs.launch
        launch_H = jobs.JobHParams()
        launch_H.name = "/".join([name] + descriptors)
        filtered_trial_bindings = []
        dry_run = False
        for k, v in trial_bindings:
            if k in launch_kwarg_keys:
                setattr(launch_H, launch_kwarg_keys[k], v)
            elif k == "dry_run":
                dry_run = v
            else:
                filtered_trial_bindings.append((k, v))

        if dry_run:
            print(f"{launch_H.name}: {filtered_trial_bindings}")
        else:
            hparams = hparam_class()
            hparams.override_from_pairs(filtered_trial_bindings)
            to_launch.append(jobs.Job(fn=partial(launch_fn, hparams), params=launch_H))
    return to_launch


def experiment_fn_launcher(experiment_dict, fn):
    def launcher(exp, name, input_path_folder, input_path_index, output_folder, **extra_args):
        try:
            trials = experiment_dict[exp]
        except KeyError:
            raise ValueError(f"Couldn't find experiment '{exp}'")

        fn(name, trials, input_path_folder, input_path_index, output_folder, **extra_args)

    return launcher


def experiment_def_launcher(experiment_dict, main_fn, **default_bindings):
    """
    Use like this:

    if __name__ == "__main__":
        fire.Fire(
            experiment_def_launcher(
                experiment_dict=experiment_definitions(),
                main_fn=train_rm.main,
            )
        )
    """

    @wraps(main_fn)
    def fn(name, trials, input_path_folder, input_path_index, output_folder, **extra_args):
        # Bind remaining extra arguments from the defaults and from the command line
        trials = combos(
            *[bind(k, v) for k, v in default_bindings.items()],
            trials,
            *[bind(k, v) for k, v in extra_args.items()],
            *[bind("input_path_folder", input_path_folder)],
            *[bind("input_path_index", input_path_index)],
            *[bind("output_folder", output_folder)]

        )

        exp_jobs = get_experiment_jobs(name, main_fn, trials,)
        return jobs.multilaunch(exp_jobs)

    return experiment_fn_launcher(experiment_dict, fn)
