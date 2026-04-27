"# Mario-LLM-" 
该系统是基于MAGB(https://github.com/sktsherlock/MAGB.git) 进行重构的，关于Mario的代码实现位于MLLM下
如果想要完整的跑完该代码,请查看 'How_to_use_MAGB.md' ，根据其说明将数据集下载到对应位置
之后阅读 'MLLM/README.md' ，使用我们添加的复现代码进行训练,训练后的模型默认放置到./trained_modals下
之后可以运行FeatureExtractor.py进行特征提取,在主目录下我们添加了run_movies_mario_lp.py和run_movies_mario_nc.py这两种利用Mario生成的嵌入运行link prediction和node classfication的代码
如果需要和MAGB中的其他模型进行横向对比,请遵照How_to_use_MAGB.md中的使用说明进行测试

@misc{yan2025graphmeetsmultimodalbenchmarking,
      title={When Graph meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graphs Learning},
      author={Hao Yan and Chaozhuo Li and Jun Yin and Zhigang Yu and Weihao Han and Mingzheng Li and Zhengxin Zeng and Hao Sun and Senzhang Wang},
      year={2025},
      eprint={2410.09132},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.09132},
}
