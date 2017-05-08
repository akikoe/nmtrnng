# NMT+RNNG: A hybrid decoder for NMT which combines the decoder of an attentin-based neural machine translation model with Recurrent Nerual Network Grammars
We have presented a novel syntax-aware NMT model in the target side, called NMT+RNNG ("[Learning to Parse and Translate Improves Neural Machine Translation](https://arxiv.org/pdf/1702.03525.pdf)" [1].
The decoder of the NMT+RNNG combines a usual conditional language model and Recurrent Neural Network Grammas (RNNGs) [2], which enables the proposed model to learn to parse and translate.

## Description
C++ implementation.

1. `NMTRNNG.xpp`: our proposed model (NMT+RNNG).
2. `AttentionBiEncDec.xpp`: Baseline ANMT model
3. `/data/`: Tanaka corpus (JP-EN) [4]

## Requirement
  * Eigen, a template libary for linear algebra (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
  * N3LP, C++ libaray for neural network-based NLP (<https://github.com/hassyGo/N3LP>) (!) Note that some implementations are not available yet when running these codes efficiently
  * Optional: SyntaxNet, a syntactic parser (<https://github.com/tensorflow/models/tree/master/syntaxnet>)

## Usage
   1. Modify the paths of `EIGEN_LOCATION` and `SHARE_LOCATION` in `Makefile`. 
   2. `$ bash setup.sh`
   3. `$ ./nmtrnng` (Then, it starts training the `NMTRNNG` model.)
   4. Modify `main.cpp` if you would like to try another model.

## References
   * [1] Akiko Eriguchi, Yoshimasa Tsuruoka, and Kyunghyun Cho. 2017. "[Learning to Parse and Translate Improves Neural Machine Translatioin](https://arxiv.org/pdf/1702.03525.pdf)". In Proceeding of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017). To appear.

   * [2] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. 2016. "[Recurrent Neural Network Grammars](https://arxiv.org/abs/1602.07776)". In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies.   
   * [3] [Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus)

## Contact
Thank you for your having interest in our work. If there are any issues, feel free to contact me.
   * eriguchi [.at.] logos.t.u-tokyo.ac.jp