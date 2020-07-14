import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem
import json

from kfp import components
from kfp.components import func_to_container_op

platform = 'GCP'

def convert_mnist_experiment_result(experiment_result, gs_bucket, epochs) -> str:
    import json
    r = json.loads(experiment_result)
    print('Before appending the results')
    args=[]
    for hp in r:
        print(hp)
        args.append(hp["name"])
        args.append(hp["value"])
        #args.append("%s=%s" % (hp["name"], hp["value"]))
    args.append('--bucket_name')
    args.append(gs_bucket)
    args.append('--epochs')
    args.append(epochs)
    print('After appending the result')
    print(args)
    value=' '.join(map(str, args))
    print('Returning')
    print(value)
    return value


@dsl.pipeline(
  name='Fashion MNIST',
  description='Fashion MNIST.'
)
def mnist_pipeline(
            experiment_name = 'mnist', 
            namespace = 'kubeflow',
            gs_bucket='gs://your-bucket/export', 
		    epochs=10, 
		    batch_size=128,
		    model_dir='gs://your-bucket/export', 
		    model_name='dummy',
		    server_name='dummy'):

    objectiveConfig = {
            "type": "maximize",
            "goal": 0.85,
            "objectiveMetricName": "accuracy",
        }
    algorithmConfig = {"algorithmName" : "random"}
    parameters = [
            {"name": "--learning_rate", "parameterType": "double", "feasibleSpace": {"min": "0.001","max": "0.003"}},
            {"name": "--batch_size", "parameterType": "discrete", "feasibleSpace": {"list": ["512", "1024", "2048"]}},
        ]
    rawTemplate = {
            "apiVersion": "kubeflow.org/v1",
            "kind": "TFJob",
            "metadata": {
                "name": "{{.Trial}}",
                "namespace": "{{.NameSpace}}"
            },
            "spec": {
                "tfReplicaSpecs": {
                  "Worker": {
                    "replicas": 1,
                    "restartPolicy": "OnFailure",
                    "template": {
                        "spec": {
                            "containers": [
                              {
                                "name": "tensorflow",
                                "image": "gcr.io/dais-data-dev-txwj/mnist/train:latest",
                                "imagePullPolicy": "Always",
                                "command": [
                                    "sh",
                                    "-c"
                                ],
                                "args": [                                
                                    "python /train.py --epochs 1 --save_model 0  {{- with .HyperParameters}} {{- range .}} {{.Name}}={{.Value}} {{- end}} {{- end}}"
                                ]
                              }
                            ]
                        }
                      }
                    }
                }
            }
    }

    trialTemplate = {
        "goTemplate": {
            "rawTemplate": json.dumps(rawTemplate)
        }
    }
   
    katib_experiment_launcher_op = components.load_component_from_url('https://raw.githubusercontent.com/kubeflow/pipelines/master/components/kubeflow/katib-launcher/component.yaml')
    katibOp = katib_experiment_launcher_op(
            experiment_name=experiment_name,
            experiment_namespace=namespace,
            parallel_trial_count=3,
            max_trial_count=12,
            objective=str(objectiveConfig),
            algorithm=str(algorithmConfig),
            trial_template=str(trialTemplate),
            parameters=str(parameters),
            delete_finished_experiment=False)


    train_args = [
  		'--bucket_name', gs_bucket, 
  		'--epochs', epochs, 
    ]

    convert_op = func_to_container_op(convert_mnist_experiment_result)
    op2 = convert_op(katibOp.output, gs_bucket, epochs)

    train = dsl.ContainerOp(
        name='train',
        image='gcr.io/dais-data-dev-txwj/mnist/train:latest',
        arguments=op2.output
    )

    serve_args = [
        '--model_path', model_dir,
        '--model_name', model_name,
        '--server_name', server_name
    ]

    serve = dsl.ContainerOp(
        name='serve',
        image='gcr.io/dais-data-dev-txwj/mnist/serve:latest',
        arguments=serve_args
    )

    steps = [katibOp, op2, train, serve]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))
        else:
            step.apply(onprem.mount_pvc(pvc_name, 'local-storage', '/mnt'))

    op2.after(katibOp)
    train.after(op2)
    serve.after(train)



if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(mnist_pipeline, __file__ + '.tar.gz')
