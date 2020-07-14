import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.onprem as onprem
import json

from kfp import components
from kfp.components import func_to_container_op

platform = 'GCP'

def step1() -> str:
    import json
    return json.dumps([{"name": "--learning_rate","value": "0.0029703918259800085"},{"name": "--batch_size","value": "512"}])

def convert_mnist_experiment_result(experiment_result, gs_bucket, epochs) -> list:
    import json
    print(experiment_result)
    print(gs_bucket)
    print(epochs)

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
    return args


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



    train_args = [
  		'--bucket_name', gs_bucket, 
  		'--epochs', epochs, 
    ]

    convert_op1 = func_to_container_op(step1)
    op1 = convert_op1()


    convert_op = func_to_container_op(convert_mnist_experiment_result)
    op2 = convert_op(op1.output, gs_bucket, epochs)

    train = dsl.ContainerOp(
        name='train',
        image='gcr.io/dais-data-dev-txwj/mnist/train:latest',
        arguments=op2.output
    )

    steps = [op1, op2, train]
    for step in steps:
        if platform == 'GCP':
            step.apply(gcp.use_gcp_secret('user-gcp-sa'))
        else:
            step.apply(onprem.mount_pvc(pvc_name, 'local-storage', '/mnt'))

    op2.after(op1)
    train.after(op2)


if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(mnist_pipeline, __file__ + '.tar.gz')
