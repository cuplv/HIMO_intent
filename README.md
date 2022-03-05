## HIMO: Logic-Based Hierarchical Intent Monitoring for Mobile Robots

HIMO is a framework that uses temporal logic specifications to infer the unknown intent of a robotic agent through passive observations of its actions over time.

<p align="center">
    <img src="./images/kitchen.png" width="70%">
</p>
<i>Intent monitoring can be key to enabling a human-robot collaboration. If the robot is aware of that chef’s intent is to repeatedly bring a pot, water, meat, in order, and then go to the cooktop, it can help the chef by bringing meats while he is going to get some water, or can avoid collisions predicting their future positions.</i>

---

## Installation
On your local computer

```bash
git clone git@github.com:cuplv/HIMO_intent.git
cd HIMO_intent
docker build . -t himo
```

To run

```bash
docker run -it --rm -p 8888:8888 -v $PWD/.:/root/HIMO_intent himo
```

Inside the container

```bash
cd /root/HIMO_intent
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

## You can test HIMO

Test HIMO on following two environments:
- a random environment in `High-level-monitor.ipynb`
- a TH&Ouml;R dataset in `thor_test.ipynb`.

### TH&Ouml;R Dataset
For more details about TH&Ouml;R Dataset, see http://thor.oru.se/.
