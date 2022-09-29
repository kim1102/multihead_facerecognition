# multihead_facerecognition

Bug report: devkim1102@gmail.com

![figure](./fig1.jpg)

## Usage example

```sh
python -m torch.distributed.run --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train_multihead.py
```

## Release History

* 0.1.1
    * Base code aploaded
