import mlflow
import os
import collections


class BaseMLFlowSetup:
    def log_metrics(self,
                    experiment_name: str,
                    args_dict=None,
                    tracking_uri: str='http://voice.orbl.io:5000',
                    **metrics):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(mlflow.get_experiment_by_name(experiment_name))
        print(experiment_name)
        print(mlflow.get_experiment(experiment_id='1'))
        mlflow.set_tag("command", os.path.realpath(__file__))

        if args_dict is not None:
            try:
                mlflow.log_params(args_dict)
            except Exception as err:
                print('Cannot add args_dict to mlflow Experiment')
                print('Passed:', args_dict)
                print('Error:', err)

        print(metrics)

        for metric_name in metrics:
            metric_val = metrics[metric_name]

            if isinstance(metric_val, collections.Iterable):
                step = 0

                for val in metric_val:
                    mlflow.log_metric(key=metric_name, value=val, step=step)
                    step += 1
            else:
                mlflow.log_metric(key=metric_name, value=metric_val)