<div align="center">
  
# Slime+ 
Slime+ is an RLVR toolkit designed based on Slime, including Gym, offline-inference, reward server, etc.

| [**AlphaXiv**](https://www.alphaxiv.org/abs/2512.23165)
| [**ArXiv**](https://www.arxiv.org/abs/2512.23165)
| [**Checkpoints**](https://huggingface.co/MikaStars39/PeRL)
| [**Wandb Log**](https://wandb.ai/mikastars-zhejiang-university/PeRL_logs)

</div>


```python
# launch the master node of ray in container
LOCAL_IP=$(hostname -I | awk '{print $1}') # get master node local ip for submitting
ray start --head --node-ip-address ${LOCAL_IP} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
```