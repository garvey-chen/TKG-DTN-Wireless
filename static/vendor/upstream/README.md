The rapid evolution of wireless networks presents unprecedented challenges in managing complex and dynamic systems. Existing methods are increasingly facing fundamental limitations in addressing these challenges. In this paper, we introduce WirelessAgent, a novel framework that harnesses large language models (LLMs) to create autonomous AI agents for diverse wireless network tasks. This framework integrates four core modules that mirror human cognitive processes: perception, memory, planning, and action. To implement it, we provide a basic usage based on agentic workflows and the LangGraph architecture. We demonstrate the effectiveness of WirelessAgent through a comprehensive case study on network slicing. The numerical results show that WirelessAgent achieves 44.4% higher bandwidth utilization than the Prompt-based method, while performing only $4.3\%$ below the Rule-based optimality. Notably, WirelessAgent delivers near-optimal network throughput across diverse network scenarios. These underscore the framework's potential for intelligent and autonomous control in future wireless networks.


## Implementation 

- The implementation steps of the WirelessAgent can be found in the [Video](https://youtu.be/4fqADkT_XMc) and the [Paper](https://arxiv.org/pdf/2505.01074).

- A GUI example of the WirelessAgent-enabled network slicing can be found in a Poe [canvas](https://poe.com/WirelessAgent-EN02).

- The implementation details are summarized below:


Step 1: Go to the OpenStreetMap website to download the "HKUST" map;

Step 2: Select an area and output the "HKUST" campus layout (HKUST_F.osm);

Step 3: Add the file path of HKUST_F.osm to the RayTracing_cqi.py;

Step 4: Add the RayTracing results to the WA_DS_KB.py for network slicing;

Step 5: Output the network slicing results.


Please stay tuned for updates, and feel free to reach out if you have any questions or need further information.


## License

WirelessAgent<sup>2</sup> is MIT-licensed. The license applies to the pre-trained models and datasets as well.

## Citation

If you find the repository is helpful to your project, please cite as follows:

```bibtex
@article{tong2025wirelessagent,
  title={Wirelessagent: Large language model agents for intelligent wireless networks},
  author={Tong, Jingwen and Guo, Wei and Shao, Jiawei and Wu, Qiong  and Li, Zijian and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2505.01074v1},
  year={2025}
}
```

## Acknowledgment

This code is done with the help of large language models, such as the Claude-3.7-Sonnet-Reasoning and DeepSeek-R1 models.

