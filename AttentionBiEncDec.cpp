#include "AttentionBiEncDec.hpp"
#include "ActFunc.hpp"
#include "Utils.hpp"
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>

/* Encoder-Decoder (EncDec.cpp) with Attention Mechanism:
   
  1-layer Bi-directional LSTM units with ``Global Attention``.

  Paper: "Effective Approaches to Attention-based Neural Machine Translation" by Luong et al. published in EMNLP2015.
  Pdf: http://arxiv.org/abs/1508.04025

  + Speedup technique proposed by Yuchen Qiao
  + biEncHalf
  + Optimizer (SGD)

*/
#define print(var)  \
  std::cout<<(var)<<std::endl

AttentionBiEncDec::AttentionBiEncDec(Vocabulary& sourceVoc_, 
				     Vocabulary& targetVoc_, 
				     std::vector<AttentionBiEncDec::Data*>& trainData_, 
				     std::vector<AttentionBiEncDec::Data*>& devData_, 
				     const int inputDim_,
				     const int hiddenEncDim_, 
				     const int hiddenDim_, 
				     const Real scale,
				     const bool useBlackOut_, 
				     const int blackOutSampleNum, 
				     const Real blackOutAlpha,
				     const AttentionBiEncDec::OPT opt_,
				     const Real clipThreshold_,
				     const int beamSize_, 
				     const int maxLen_,
				     const int miniBatchSize_, 
				     const int threadNum_, 
				     const Real learningRate_, 
				     const bool isTest_,
				     const int startIter_,
				     const std::string& saveDirName_):
opt(opt_),
  sourceVoc(sourceVoc_), 
  targetVoc(targetVoc_),
  trainData(trainData_), 
  devData(devData_),
  inputDim(inputDim_),
  hiddenEncDim(hiddenEncDim_), 
  hiddenDim(hiddenDim_), 
  useBlackOut(useBlackOut_),
  clipThreshold(clipThreshold_),
  beamSize(beamSize_), 
  maxLen(maxLen_), 
  miniBatchSize(miniBatchSize_), 
  threadNum(threadNum_), 
  learningRate(learningRate_), 
  isTest(isTest_),
  startIter(startIter_),
  saveDirName(saveDirName_)
{
  // this->rnd = Rand(this->rnd.next()); // (!) TODO: For Ensemble

  // LSTM units
  this->enc = LSTM(inputDim, hiddenEncDim); // Encoder
  this->enc.init(this->rnd, scale);
  this->encRev = LSTM(inputDim, hiddenEncDim); // Encoder (Reverse)
  this->encRev.init(this->rnd, scale);

  this->dec = LSTM(inputDim, hiddenDim, hiddenDim); // Decoder
  this->dec.init(this->rnd, scale);

  // LSTMs' biases set to 1 
  this->enc.bf.fill(1.0);
  this->encRev.bf.fill(1.0);
  this->dec.bf.fill(1.0);

  // Affine
  this->decInitAffine = Affine(hiddenEncDim*2, hiddenDim);
  this->decInitAffine.act = Affine::TANH;
  this->decInitAffine.init(this->rnd, scale);

  this->stildeAffine = Affine(hiddenDim + hiddenEncDim*2, hiddenDim);
  this->stildeAffine.act = Affine::TANH;
  this->stildeAffine.init(this->rnd, scale);

  // Embedding matrices
  this->sourceEmbed = MatD(inputDim, this->sourceVoc.tokenList.size());
  this->targetEmbed = MatD(inputDim, this->targetVoc.tokenList.size());
  this->rnd.uniform(this->sourceEmbed, scale);
  this->rnd.uniform(this->targetEmbed, scale);

  this->Wgeneral = MatD(hiddenDim, hiddenEncDim*2); // attenType == GENERAL; for Sequence
  this->rnd.uniform(this->Wgeneral, scale);

  // Softmax / BlackOut
  if (!this->useBlackOut) {
    this->softmax = SoftMax(hiddenDim, this->targetVoc.tokenList.size());
  } else {
    VecD freq = VecD(this->targetVoc.tokenList.size());
    for (int i = 0; i < (int)this->targetVoc.tokenList.size(); ++i) {
      freq.coeffRef(i, 0) = this->targetVoc.tokenList[i]->count;
    }
    this->blackOut = BlackOut(hiddenDim, this->targetVoc.tokenList.size(), blackOutSampleNum);
    this->blackOut.initSampling(freq, blackOutAlpha);
  }

  this->zeros = VecD::Zero(hiddenDim); // Zero vector
  this->zerosEnc = VecD::Zero(this->hiddenEncDim); // Zero vector
  this->zeros2 = VecD::Zero(this->hiddenEncDim*2); // Zero vector

  /* For automatic tuning */
  this->prevPerp = REAL_MAX;
  // this->prevModelFileName = this->saveModel(-1);
}

void AttentionBiEncDec::biEncode(const AttentionBiEncDec::Data* data,
				 AttentionBiEncDec::ThreadArg& arg,
				 const bool train){ // Encoder for sequence
  int length = data->src.size()-1; // For word

  for (int i = 0; i < arg.srcLen; ++i) {
    if (i == 0) {
      this->enc.forward(this->sourceEmbed.col(data->src[i]), arg.encState[i]);
      this->encRev.forward(this->sourceEmbed.col(data->src[length-i]), arg.encRevState[i]);
    } else {
      this->enc.forward(this->sourceEmbed.col(data->src[i]), arg.encState[i-1], arg.encState[i]);
      this->encRev.forward(this->sourceEmbed.col(data->src[length-i]), arg.encRevState[i-1], arg.encRevState[i]);
    }
    arg.biEncState[i].segment(0, this->hiddenEncDim).noalias() = arg.encState[i]->h;
    arg.biEncState[length-i].segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[i]->h;

    if (train) {
      arg.encState[i]->delc = this->zerosEnc; // (!) Initialize here for backward
      arg.encState[i]->delh = this->zerosEnc;
      arg.encRevState[i]->delc = this->zerosEnc;
      arg.encRevState[i]->delh = this->zerosEnc;
    }
  }
  // Affine
  arg.decInitStateEnd.segment(0, this->hiddenEncDim).noalias() = arg.encState[arg.srcLen-1]->h;
  arg.decInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[arg.srcLen-1]->h; 
}

void AttentionBiEncDec::biEncoderBackward2(const AttentionBiEncDec::Data* data,
					   AttentionBiEncDec::ThreadArg& arg,
					   AttentionBiEncDec::Grad& grad) {
  // Backward (this->enc)
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXF);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXO);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, this->sourceEmbed.col(data->src[i]), LSTM::WXU);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->enc.backward2(arg.encState[i], grad.lstmSrcGrad, arg.encState[i-1]->h, LSTM::WHU);
  }
  // Backward (this->encRev)
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXF);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXO);
  }
  for (int i = arg.srcLen-1; i >= 0; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, this->sourceEmbed.col(data->src[arg.srcLen-i-1]), LSTM::WXU);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.srcLen-1; i >= 1; --i){
    this->encRev.backward2(arg.encRevState[i], grad.lstmSrcRevGrad, arg.encRevState[i-1]->h, LSTM::WHU);
  }
}

void AttentionBiEncDec::decoder(AttentionBiEncDec::ThreadArg& arg,
				std::vector<LSTM::State*>& decState, 
				VecD& s_tilde, 
				const int tgtNum,
				const int i, 
				const bool train) {
  if (i == 0) { // initialize decoder's initial state
    this->decInitAffine.forward(arg.decInitStateEnd, arg.decState[i]->h);
    arg.decState[i]->c = this->zeros;
  } else { // i >= 1 
    // input-feeding approach [Luong et al., EMNLP2015]
    this->dec.forward(this->targetEmbed.col(tgtNum), s_tilde,
		      decState[i-1], decState[i]); // (xt, at (use previous ``s_tidle``), prev, cur)
  }
  if (train) {
    arg.decState[i]->delc = this->zeros;
    arg.decState[i]->delh = this->zeros;
    arg.decState[i]->dela = this->zeros;
  }
}

void AttentionBiEncDec::decoderBackward2(const AttentionBiEncDec::Data* data,
					 AttentionBiEncDec::ThreadArg& arg,
					 AttentionBiEncDec::Grad& grad) {
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXI);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXF);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXO);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), LSTM::WXU);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHI);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHF);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHO);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.decState[i-1]->h, LSTM::WHU);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAI);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAF);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAO);
  }
  for (int i = arg.tgtLen-1; i >= 1; --i){
    this->dec.backward2(arg.decState[i], grad.lstmTgtGrad, arg.s_tilde[i-1], LSTM::WAU);
  }
}

void AttentionBiEncDec::decoderAttention(AttentionBiEncDec::ThreadArg& arg, 
					 const LSTM::State* decState,
					 VecD& contextSeq, 
					 VecD& s_tilde, 
					 VecD& stildeEnd) { // CalcLoss / Test
  /* Attention */
  // sequence
  contextSeq = this->zeros2;

  this->calculateAlphaBiEnc(arg, decState);

  for (int j = 0; j < arg.srcLen; ++j) {
    contextSeq.noalias() += arg.alphaSeqVec.coeff(j, 0) * arg.biEncState[j];
  }

  stildeEnd.segment(0, this->hiddenDim).noalias() = decState->h;
  stildeEnd.segment(this->hiddenDim, this->hiddenEncDim*2).noalias() = contextSeq;

  this->stildeAffine.forward(stildeEnd, s_tilde);
}

void AttentionBiEncDec::decoderAttention(AttentionBiEncDec::ThreadArg& arg,
					 const int i,
					 const bool train) { // Train
  /* Attention */
  // sequence
  arg.contextSeqList[i] = this->zeros2;

  this->calculateAlphaBiEnc(arg, arg.decState[i], i);

  for (int j = 0; j < arg.srcLen; ++j) {
    arg.contextSeqList[i].noalias() += arg.alphaSeq.coeff(j, i) * arg.biEncState[j];
  }
  arg.stildeEnd[i].segment(0, this->hiddenDim).noalias() = arg.decState[i]->h;
  arg.stildeEnd[i].segment(this->hiddenDim, this->hiddenEncDim*2).noalias() = arg.contextSeqList[i];

  this->stildeAffine.forward(arg.stildeEnd[i], arg.s_tilde[i]);
}

struct sort_pred {
  bool operator()(const AttentionBiEncDec::DecCandidate left, const AttentionBiEncDec::DecCandidate right) {
    return left.score > right.score;
  }
};

void AttentionBiEncDec::translate(AttentionBiEncDec::Data* data, 
				  AttentionBiEncDec::ThreadArg& arg,
				  std::vector<int>& translation,
				  const bool train) {
  const Real minScore = -1.0e+05;
  const int maxLength = this->maxLen;
  const int beamSize = arg.candidate.size();
  int showNum;
  if ((int)arg.candidate.size() > 1) {
    showNum = 5;
  } else {
    showNum = 1;
  }
  MatD score = MatD(this->targetEmbed.cols(), beamSize);
  std::vector<AttentionBiEncDec::DecCandidate> candidateTmp(beamSize);

  for (auto it = arg.candidate.begin(); it != arg.candidate.end(); ++it){
    it->init();
  }
  arg.init(*this, data, false);
  this->biEncode(data, arg, false); // encoder

  for (int i = 0; i < maxLength; ++i) {
    for (int j = 0; j < beamSize; ++j) {
      VecD stildeEnd = VecD(this->hiddenDim + this->hiddenEncDim*2);

      if (arg.candidate[j].stop) {
	score.col(j).fill(arg.candidate[j].score);
	continue;
      }
      if (i == 0) {
	VecD decInitStateEnd = VecD(this->hiddenEncDim*2);
	decInitStateEnd.segment(0, this->hiddenEncDim).noalias() = arg.encState[arg.srcLen-1]->h;
	decInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim).noalias() = arg.encRevState[arg.srcLen-1]->h;
	this->decInitAffine.forward(decInitStateEnd, arg.candidate[j].curDec.h);
	arg.candidate[j].curDec.c = this->zeros;
      } else {
	arg.candidate[j].prevDec.h = arg.candidate[j].curDec.h;
	arg.candidate[j].prevDec.c = arg.candidate[j].curDec.c;
	this->dec.forward(this->targetEmbed.col(arg.candidate[j].generatedTarget[i-1]), arg.candidate[j].s_tilde,
			  &arg.candidate[j].prevDec, &arg.candidate[j].curDec);
      }
      this->decoderAttention(arg, &arg.candidate[j].curDec, arg.candidate[j].contextSeq, 
			     arg.candidate[j].s_tilde, stildeEnd);
      if (!this->useBlackOut) {
	this->softmax.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      } else {
	this->blackOut.calcDist(arg.candidate[j].s_tilde, arg.candidate[j].targetDist);
      }
      score.col(j).array() = arg.candidate[j].score + arg.candidate[j].targetDist.array().log();
    }
    for (int j = 0, row, col; j < beamSize; ++j) {
      score.maxCoeff(&row, &col); // Greedy
      candidateTmp[j] = arg.candidate[col];
      candidateTmp[j].score = score.coeff(row, col);

      if (candidateTmp[j].stop) { // if "EOS" comes up...
	score.col(col).fill(minScore);
	continue;
      }

      candidateTmp[j].generatedTarget.push_back(row);

      if (row == this->targetVoc.eosIndex) {
	candidateTmp[j].stop = true;
      }

      if (i == 0) {
	score.row(row).fill(minScore);
      } else {
	score.coeffRef(row, col) = minScore;
      }
    }

    arg.candidate = candidateTmp;

    std::sort(arg.candidate.begin(), arg.candidate.end(), sort_pred());

    if (arg.candidate[0].generatedTarget.back() == this->targetVoc.eosIndex) {
      break;
    }
  }

  if (train) {
    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " "; 
    }
    std::cout << std::endl;

    for (int i = 0; i < showNum; ++i) {
      std::cout << i+1 << " (" << arg.candidate[i].score << "): ";

      for (auto it = arg.candidate[i].generatedTarget.begin(); it != arg.candidate[i].generatedTarget.end(); ++it) {
	std::cout << this->targetVoc.tokenList[*it]->str << " ";
      }
      std::cout << std::endl;
    }

    for (auto it = data->src.begin(); it != data->src.end(); ++it) {
      std::cout << this->sourceVoc.tokenList[*it]->str << " ";
    }
    std::cout << std::endl;
  } else {
    this->makeTrans(arg.candidate[0].generatedTarget, data->trans);
  }
}

void AttentionBiEncDec::train(AttentionBiEncDec::Data* data, 
			      AttentionBiEncDec::ThreadArg& arg,
			      AttentionBiEncDec::Grad& grad,
			      const bool train = true) { // mini-batchsize=1の学習 w/ inputFeeding
  int length = data->src.size()-1; // source words

  arg.init(*this, data, train);
  this->biEncode(data, arg, train); // encoder

  for (int i = 0; i < arg.tgtLen; ++i) {
    // 1) Let a decoder run forward for 1 step; PUSH
    this->decoder(arg, arg.decState, arg.s_tilde[i-1], data->tgt[i-1], i, train);
    /* Attention */
    this->decoderAttention(arg, i, train);
  }

  // Backward
  if (!this->useBlackOut) {
    for (int i = 0; i < arg.tgtLen; ++i) {
      this->softmax.calcDist(arg.s_tilde[i], arg.targetDist);
      arg.loss += this->softmax.calcLoss(arg.targetDist, data->tgt[i]);
      this->softmax.backward(arg.s_tilde[i], arg.targetDist, data->tgt[i], 
			     arg.del_stilde[i], grad.softmaxGrad);
    }
  } else { // Share the negative samples
    this->blackOut.sampling2(arg.blackOutState[0], this->targetVoc.unkIndex); // unk
    for (int i = 0; i < arg.tgtLen; ++i) {
      // word prediction
      arg.blackOutState[0].sample[0] = data->tgt[i];
      arg.blackOutState[0].weight.col(0) = this->blackOut.weight.col(data->tgt[i]);
      arg.blackOutState[0].bias.coeffRef(0, 0) = this->blackOut.bias.coeff(data->tgt[i], 0);
      this->blackOut.calcSampledDist2(arg.s_tilde[i], arg.targetDistVec[i], arg.blackOutState[0]);
      arg.loss += this->blackOut.calcSampledLoss(arg.targetDistVec[i]);
    }
    this->blackOut.backward_1(arg.tgtLen, data->tgt, arg.targetDistVec, arg.blackOutState, arg.del_stilde);
    this->blackOut.backward_2(arg.tgtLen, data->tgt, arg.s_tilde, arg.blackOutState, grad.blackOutGrad);
  }  

  /* -- Backpropagation starts -- */
  for (int i = arg.tgtLen-1; i >= 0; --i) {
    if (i < arg.tgtLen-1) {
      arg.del_stilde[i].noalias() += arg.decState[i+1]->dela;
      // add gradients to the previous del_stilde 
      // by input-feeding [Luong et al., EMNLP2015]
    } else {}
      
    this->stildeAffine.backward1(arg.s_tilde[i], arg.del_stilde[i], arg.del_stilde[i], 
				 arg.del_stildeEnd, grad.stildeAffineGrad);
    arg.decState[i]->delh.noalias() += arg.del_stildeEnd.segment(0, this->hiddenDim);
  
    // del_contextSeq
    for (int j = 0; j < arg.srcLen; ++j) { // Seq
      arg.del_alphaSeqTmp = arg.alphaSeq.coeff(j,i) * arg.del_stildeEnd.segment(this->hiddenDim, this->hiddenEncDim*2);
      arg.encState[j]->delh.noalias() += arg.del_alphaSeqTmp.segment(0, this->hiddenEncDim);
      arg.encRevState[arg.srcLen-j-1]->delh.noalias() += arg.del_alphaSeqTmp.segment(this->hiddenEncDim, this->hiddenEncDim);
      arg.del_alphaSeq.coeffRef(j, 0) = arg.del_stildeEnd.segment(this->hiddenDim, this->hiddenEncDim*2).dot(arg.biEncState[j]);
    }
    arg.del_alignScore = arg.alphaSeq.col(i).array()*(arg.del_alphaSeq.array()-arg.alphaSeq.col(i).dot(arg.del_alphaSeq)); // X.array() - scalar; np.array() -= 1

    for (int j = 0; j < arg.srcLen; ++j) {
      arg.del_WgeneralTmp = this->Wgeneral.transpose()*arg.decState[i]->h;
      arg.encState[j]->delh.noalias() += arg.del_WgeneralTmp.segment(0, this->hiddenEncDim) * arg.del_alignScore.coeff(j, 0);
      arg.encRevState[arg.srcLen-j-1]->delh.noalias() += arg.del_WgeneralTmp.segment(this->hiddenEncDim, this->hiddenEncDim) * arg.del_alignScore.coeff(j, 0);

      arg.decState[i]->delh.noalias() += (this->Wgeneral*arg.biEncState[j])*arg.del_alignScore.coeff(j, 0);
      grad.Wgeneral += arg.del_alignScore.coeff(j, 0)*arg.decState[i]->h*arg.biEncState[j].transpose();
    }
    if (i > 0) {
      // Backward
      this->dec.backward1(arg.decState[i-1], arg.decState[i], 
			  grad.lstmTgtGrad, this->targetEmbed.col(data->tgt[i-1]), arg.s_tilde[i-1]);
      if (grad.targetEmbed.count(data->tgt[i-1])) {
	grad.targetEmbed.at(data->tgt[i-1]) += arg.decState[i]->delx;
      } else {
	grad.targetEmbed[data->tgt[i-1]] = arg.decState[i]->delx;
      }
    } else {}
  }
  for (int d = 0, D = arg.stildeEnd[0].rows(); d < D; ++d){
    for (int i = 0; i < arg.tgtLen; ++i) {
      this->stildeAffine.backward2(arg.stildeEnd[i], arg.del_stilde[i], d, grad.stildeAffineGrad);
    }
  }

  // Decoder (this->dec)
  this->decoderBackward2(data, arg, grad);
  
  // Decoder -> Encoder
  this->decInitAffine.backward(arg.decInitStateEnd, arg.decState[0]->h, arg.decState[0]->delh, arg.del_decInitStateEnd, grad.decInitAffineGrad);
  arg.encState[arg.srcLen-1]->delh.noalias() += arg.del_decInitStateEnd.segment(0, this->hiddenEncDim);
  arg.encRevState[arg.srcLen-1]->delh.noalias() += arg.del_decInitStateEnd.segment(this->hiddenEncDim, this->hiddenEncDim);

  for (int i = arg.srcLen-1; i >= 0; --i) {
    if (i == 0 ) {
      this->enc.backward1(arg.encState[i], grad.lstmSrcGrad, 
			  this->sourceEmbed.col(data->src[i]));
      this->encRev.backward1(arg.encRevState[i], grad.lstmSrcRevGrad, 
			    this->sourceEmbed.col(data->src[length])); // length+1-1 = lengthHS -i
    } else {
      this->enc.backward1(arg.encState[i-1], arg.encState[i], grad.lstmSrcGrad, 
			  this->sourceEmbed.col(data->src[i]));
      this->encRev.backward1(arg.encRevState[i-1], arg.encRevState[i], grad.lstmSrcRevGrad, 
			     this->sourceEmbed.col(data->src[arg.srcLen-i-1]));
    }
    if (grad.sourceEmbed.count(data->src[i])) {
      grad.sourceEmbed.at(data->src[i]).noalias() += arg.encState[i]->delx;
    } else {
      grad.sourceEmbed[data->src[i]].noalias() = arg.encState[i]->delx;
    }
    if (grad.sourceEmbed.count(data->src[arg.srcLen-i-1])) {
      grad.sourceEmbed.at(data->src[arg.srcLen-i-1]).noalias() += arg.encRevState[i]->delx;
    } else {
      grad.sourceEmbed[data->src[arg.srcLen-i-1]].noalias() = arg.encRevState[i]->delx;
    }
  }
  
  // Encoder (this->enc; this->encRev)
  this->biEncoderBackward2(data, arg, grad);
}       

void AttentionBiEncDec::calculateAlphaBiEnc(AttentionBiEncDec::ThreadArg& arg,
					    const LSTM::State* decState) { // calculate attentional weight;
  for (int i = 0; i < arg.srcLen; ++i) {
    arg.alphaSeqVec.coeffRef(i, 0) = decState->h.dot(this->Wgeneral * arg.biEncState[i]);
  }

  // softmax of ``alphaSeq``
  arg.alphaSeqVec.array() -= arg.alphaSeqVec.maxCoeff(); // stable softmax
  arg.alphaSeqVec = arg.alphaSeqVec.array().exp(); // exp() operation for all elements; np.exp(alphaSeq) 
  arg.alphaSeqVec /= arg.alphaSeqVec.array().sum(); // alphaSeq.sum()
}

void AttentionBiEncDec::calculateAlphaBiEnc(AttentionBiEncDec::ThreadArg& arg, 
					    const LSTM::State* decState, 
					    const int colNum) { // calculate attentional weight;
  for (int i = 0; i < arg.srcLen; ++i) {
    arg.alphaSeq.coeffRef(i, colNum) = decState->h.dot(this->Wgeneral * arg.biEncState[i]);
  }

  // softmax of ``alphaSeq``
  arg.alphaSeq.col(colNum).array() -= arg.alphaSeq.col(colNum).maxCoeff(); // stable softmax
  arg.alphaSeq.col(colNum) = arg.alphaSeq.col(colNum).array().exp(); // exp() operation for all elements; np.exp(alphaSeq) 
  arg.alphaSeq.col(colNum) /= arg.alphaSeq.col(colNum).array().sum(); // alphaSeq.sum()
}

bool AttentionBiEncDec::trainOpenMP(AttentionBiEncDec::Grad& grad) {
  static std::vector<AttentionBiEncDec::ThreadArg> args;
  static std::vector<std::pair<int, int> > miniBatch;
  static std::vector<AttentionBiEncDec::Grad> grads;

  Real lossTrain = 0.0;
  Real lossDev = 0.0;
  Real tgtNum = 0.0;
  Real gradNorm;
  Real lr = this->learningRate;
  static float countModel = this->startIter-0.5;

  float countModelTmp = countModel;
  std::string modelFileNameTmp = "";

  if (args.empty()) {
    grad = AttentionBiEncDec::Grad(*this);
    for (int i = 0; i < this->threadNum; ++i) {
      args.push_back(AttentionBiEncDec::ThreadArg(*this));
      grads.push_back(AttentionBiEncDec::Grad(*this));
    }
    for (int i = 0, step = this->trainData.size()/this->miniBatchSize; i< step; ++i) {
      miniBatch.push_back(std::pair<int, int>(i*this->miniBatchSize, (i == step-1 ? this->trainData.size()-1 : (i+1)*this->miniBatchSize-1)));
      // Create pairs of MiniBatch, e.g. [(0,3), (4, 7), ...]
    }
  }

  auto startTmp = std::chrono::system_clock::now();
  this->rnd.shuffle(this->trainData);

  int count = 0;
  for (auto it = miniBatch.begin(); it != miniBatch.end(); ++it) {
    std::cout << "\r"
	      << "Progress: " << ++count << "/" << miniBatch.size() << " mini batches" << std::flush;

    for (auto it = args.begin(); it != args.end(); ++it) {
      it->initLoss();
    }
#pragma omp parallel for num_threads(this->threadNum) schedule(dynamic) shared(args, grad, grads)
    for (int i = it->first; i <= it->second; ++i) {
      int id = omp_get_thread_num();

      this->train(this->trainData[i], args[id], grads[id]);

      /* ..Gradient Checking.. :) */
      // this->gradientChecking(this->trainData[i], args[id], grads[id]); 
    }
    for (int id = 0; id < this->threadNum; ++id) {
      grad += grads[id];
      grads[id].init();
      lossTrain += args[id].loss;
      args[id].loss = 0.0;
    }

    gradNorm = sqrt(grad.norm())/this->miniBatchSize;
    if (Utils::infNan2(gradNorm)) {
      countModel = countModelTmp;

      grad.init();
      std::cout << "(!) Error: INF/ NAN Gradients. Resume the training." << std::endl;
      
      if (modelFileNameTmp != "") {
        int systemVal = system(((std::string)"rm " + modelFileNameTmp).c_str());
	if (systemVal == -1) {
	  std::cout << "Fails to remove "
		    << modelFileNameTmp.c_str() << std::endl;
	}
      }
      return false;
    }
    lr = (gradNorm > this->clipThreshold ? this->clipThreshold*this->learningRate/gradNorm : this->learningRate);
    lr /= this->miniBatchSize;

    if (this->opt == AttentionBiEncDec::SGD) {
      // Update the gradients by SGD
      grad.sgd(*this, lr);
    }
    grad.init();

    if (count == ((int)miniBatch.size()/2)) { // saveModel after halving epoch
      this->saveModel(grad, countModel);
      countModel += 1.;
    }

  }
  // Save a model
  std::string currentModelFileName;
  std::string currentGradFileName;
  std::tie(currentModelFileName, currentGradFileName) = this->saveModel(grad, countModel-0.5);

  std::cout << std::endl;
  auto endTmp = std::chrono::system_clock::now();
  std::cout << "Training time for this epoch: " 
	    << (std::chrono::duration_cast<std::chrono::seconds>(endTmp-startTmp).count())/60.0 << "min." << std::endl;
  std::cout << "Training Loss (/sentence):    " 
	    << lossTrain/this->trainData.size() << std::endl;

  startTmp = std::chrono::system_clock::now();
#pragma omp parallel for num_threads(this->threadNum)
  for (int i = 0; i < (int)this->devData.size(); ++i) {
    Real loss;
    int id = omp_get_thread_num();
    loss = this->calcLoss(this->devData[i], args[id], false);
#pragma omp critical
    {
      lossDev += loss;
      tgtNum += this->devData[i]->tgt.size();
    }
  }
  endTmp = std::chrono::system_clock::now();

  std::cout << "Evaluation time for this epoch: " 
	    << (std::chrono::duration_cast<std::chrono::seconds>(endTmp-startTmp).count())/60.0 
	    << "min." << std::endl;
  std::cout << "Development Perplexity and Loss (/sentence):  " 
	    << exp(lossDev/tgtNum) << ", "
	    << lossDev/this->devData.size() << std::endl;

  Real devPerp = exp(lossDev/tgtNum);
  if (this->prevPerp < devPerp){
    countModel = countModelTmp;
    std::cout << "(!) Notes: Dev perplexity became worse, Resume the training!" << std::endl;

    // system(((std::string)"rm "+modelFileNameTmp).c_str());
    // system(((std::string)"rm "+currentModelFileName).c_str());

    return false;
  }
  this->prevPerp = devPerp;
  this->prevModelFileName = currentModelFileName;
  this->prevGradFileName = currentGradFileName;

  saveResult(lossTrain/this->trainData.size(), ".trainLoss"); // Training Loss
  saveResult(exp(lossDev/tgtNum), ".devPerp");                // Perplexity
  saveResult(lossDev/this->devData.size(), ".devLoss");       // Development Loss
  
  return true;
}

Real AttentionBiEncDec::calcLoss(AttentionBiEncDec::Data* data, 
				 AttentionBiEncDec::ThreadArg& arg, 
				 const bool train) {
  Real loss = 0.0;

  arg.init(*this, data, false);
  this->biEncode(data, arg, false);

  for (int i = 0; i < arg.tgtLen; ++i) {
    this->decoder(arg, arg.decState, arg.s_tilde[i-1], data->tgt[i-1], i, false);
    this->decoderAttention(arg, arg.decState[i], arg.contextSeqList[i], arg.s_tilde[i], arg.stildeEnd[i]);
    if (!this->useBlackOut) {
      this->softmax.calcDist(arg.s_tilde[i], arg.targetDist);
      loss += this->softmax.calcLoss(arg.targetDist, data->tgt[i]);
    } else {
      if (train) {
	// word prediction
	arg.blackOutState[0].sample[0] = data->tgt[i];
	arg.blackOutState[0].weight.col(0) = this->blackOut.weight.col(data->tgt[i]);
	arg.blackOutState[0].bias.coeffRef(0, 0) = this->blackOut.bias.coeff(data->tgt[i], 0);

	this->blackOut.calcSampledDist2(arg.s_tilde[i], arg.targetDist, arg.blackOutState[0]);
	loss += this->blackOut.calcSampledLoss(arg.targetDist); // Softmax
      } else { // Test Time
	this->blackOut.calcDist(arg.s_tilde[i], arg.targetDist); //Softmax
	loss += this->blackOut.calcLoss(arg.targetDist, data->tgt[i]); // Softmax
      }
    }
  }

  return loss;
}

void AttentionBiEncDec::gradientChecking(AttentionBiEncDec::Data* data,
					 AttentionBiEncDec::ThreadArg& arg,
					 AttentionBiEncDec::Grad& grad) {
					  
  print("--Softmax");
  if (!this->useBlackOut) {
    print(" softmax_W");
    this->gradChecker(data, arg, this->softmax.weight, grad.softmaxGrad.weight);
    print(" softmax_b");
    this->gradChecker(data, arg, this->softmax.bias, grad.softmaxGrad.bias);
  } else {}

  // Decoder
  print("--Decoder");
  print(" stildeAffine_W");
  this->gradChecker(data, arg, this->stildeAffine.weight, grad.stildeAffineGrad.weightGrad);
  print(" stildeAffine_b");
  this->gradChecker(data, arg, this->stildeAffine.bias, grad.stildeAffineGrad.biasGrad);

  print(" Wgeneral");
  this->gradChecker(data, arg, this->Wgeneral, grad.Wgeneral);

  print(" dec_Wx");
  this->gradChecker(data, arg, this->dec.Wxi, grad.lstmTgtGrad.Wxi);
  this->gradChecker(data, arg, this->dec.Wxf, grad.lstmTgtGrad.Wxf);
  this->gradChecker(data, arg, this->dec.Wxo, grad.lstmTgtGrad.Wxo);
  this->gradChecker(data, arg, this->dec.Wxu, grad.lstmTgtGrad.Wxu);
  print(" dec_Wh");
  this->gradChecker(data, arg, this->dec.Whi, grad.lstmTgtGrad.Whi);
  this->gradChecker(data, arg, this->dec.Whf, grad.lstmTgtGrad.Whf);
  this->gradChecker(data, arg, this->dec.Who, grad.lstmTgtGrad.Who);
  this->gradChecker(data, arg, this->dec.Whu, grad.lstmTgtGrad.Whu);
  print(" dec_Wa");
  this->gradChecker(data, arg, this->dec.Wai, grad.lstmTgtGrad.Wai);
  this->gradChecker(data, arg, this->dec.Waf, grad.lstmTgtGrad.Waf);
  this->gradChecker(data, arg, this->dec.Wao, grad.lstmTgtGrad.Wao);
  this->gradChecker(data, arg, this->dec.Wau, grad.lstmTgtGrad.Wau);
  print(" dec_b");
  this->gradChecker(data, arg, this->dec.bi, grad.lstmTgtGrad.bi);
  this->gradChecker(data, arg, this->dec.bf, grad.lstmTgtGrad.bf);
  this->gradChecker(data, arg, this->dec.bo, grad.lstmTgtGrad.bo);
  this->gradChecker(data, arg, this->dec.bu, grad.lstmTgtGrad.bu);

  print("--Initial Decoder");
  print(" decInitAffine_W");
  this->gradChecker(data, arg, this->decInitAffine.weight, grad.decInitAffineGrad.weightGrad);
  print(" decInitAffine_b");
  this->gradChecker(data, arg, this->decInitAffine.bias, grad.decInitAffineGrad.biasGrad);

  print("--Encoder");
  print(" enc_Wx");
  this->gradChecker(data, arg, this->enc.Wxi, grad.lstmSrcGrad.Wxi);
  this->gradChecker(data, arg, this->enc.Wxf, grad.lstmSrcGrad.Wxf);
  this->gradChecker(data, arg, this->enc.Wxo, grad.lstmSrcGrad.Wxo);
  this->gradChecker(data, arg, this->enc.Wxu, grad.lstmSrcGrad.Wxu);
  print(" enc_Wh");
  this->gradChecker(data, arg, this->enc.Whi, grad.lstmSrcGrad.Whi);
  this->gradChecker(data, arg, this->enc.Whf, grad.lstmSrcGrad.Whf);
  this->gradChecker(data, arg, this->enc.Who, grad.lstmSrcGrad.Who);
  this->gradChecker(data, arg, this->enc.Whu, grad.lstmSrcGrad.Whu);
  print(" enc_b");
  this->gradChecker(data, arg, this->enc.bi, grad.lstmSrcGrad.bi);
  this->gradChecker(data, arg, this->enc.bf, grad.lstmSrcGrad.bf);
  this->gradChecker(data, arg, this->enc.bo, grad.lstmSrcGrad.bo);
  this->gradChecker(data, arg, this->enc.bu, grad.lstmSrcGrad.bu);

  print(" encRev_Wx");
  this->gradChecker(data, arg, this->encRev.Wxi, grad.lstmSrcRevGrad.Wxi);
  this->gradChecker(data, arg, this->encRev.Wxf, grad.lstmSrcRevGrad.Wxf);
  this->gradChecker(data, arg, this->encRev.Wxo, grad.lstmSrcRevGrad.Wxo);
  this->gradChecker(data, arg, this->encRev.Wxu, grad.lstmSrcRevGrad.Wxu);
  print(" encRev_Wh");
  this->gradChecker(data, arg, this->encRev.Whi, grad.lstmSrcRevGrad.Whi);
  this->gradChecker(data, arg, this->encRev.Whf, grad.lstmSrcRevGrad.Whf);
  this->gradChecker(data, arg, this->encRev.Who, grad.lstmSrcRevGrad.Who);
  this->gradChecker(data, arg, this->encRev.Whu, grad.lstmSrcRevGrad.Whu);
  print(" encRev_b");
  this->gradChecker(data, arg, this->encRev.bi, grad.lstmSrcRevGrad.bi);
  this->gradChecker(data, arg, this->encRev.bf, grad.lstmSrcRevGrad.bf);
  this->gradChecker(data, arg, this->encRev.bo, grad.lstmSrcRevGrad.bo);
  this->gradChecker(data, arg, this->encRev.bu, grad.lstmSrcRevGrad.bu);

  // Embeddings
  print("--Embedding vectors");
  print(" sourceEmbed; targetEmbed");
  this->gradChecker(data, arg, grad);
}

void AttentionBiEncDec::gradChecker(AttentionBiEncDec::Data* data,
				    AttentionBiEncDec::ThreadArg& arg, 
				    MatD& param, 
				    const MatD& grad) {
  const Real EPS = 1.0e-04;

  for (int i = 0; i < param.rows(); ++i) {
    for (int j = 0; j < param.cols(); ++j) {
      Real val= 0.0;
      Real objFuncPlus = 0.0;
      Real objFuncMinus = 0.0;
      val = param.coeff(i, j); // Θ_i
      param.coeffRef(i, j) = val + EPS;
      objFuncPlus = this->calcLoss(data, arg, true);      
      param.coeffRef(i, j) = val - EPS;
      objFuncMinus = this->calcLoss(data, arg, true);
      param.coeffRef(i, j) = val;

      Real gradVal = grad.coeff(i, j);
      Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-06) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }
}

void AttentionBiEncDec::gradChecker(AttentionBiEncDec::Data* data,
				    AttentionBiEncDec::ThreadArg& arg, 
				    VecD& param, 
				    const MatD& grad) {
  const Real EPS = 1.0e-04;

  for (int i = 0; i < param.rows(); ++i) {
    Real val= 0.0;
    Real objFuncPlus = 0.0;
    Real objFuncMinus = 0.0;
    val = param.coeff(i, 0); // Θ_i
    param.coeffRef(i, 0) = val + EPS;
    objFuncPlus = this->calcLoss(data, arg, true);      
    param.coeffRef(i, 0) = val - EPS;
    objFuncMinus = this->calcLoss(data, arg, true);
    param.coeffRef(i, 0) = val;
 
    Real gradVal = grad.coeff(i, 0);
    Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
    if ((gradVal - enumVal) > 1.0e-05) {
      std::cout << "Grad: " << gradVal << std::endl;
      std::cout << "Enum: " << enumVal << std::endl;
    } else {}
  }
}

void AttentionBiEncDec::gradChecker(AttentionBiEncDec::Data* data,
				    AttentionBiEncDec::ThreadArg& arg, 
				    AttentionBiEncDec::Grad& grad) {
  const Real EPS = 1.0e-04;

  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val = 0.0;
      Real objFuncPlus = 0.0;
      Real objFuncMinus = 0.0;
      val = this->sourceEmbed.coeff(i, it->first); // Θ_i
      this->sourceEmbed.coeffRef(i, it->first) = val + EPS;
      objFuncPlus = this->calcLoss(data, arg, true);
      this->sourceEmbed.coeffRef(i, it->first) = val - EPS;
      objFuncMinus = this->calcLoss(data, arg, true);
      this->sourceEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }

  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it) {
    for (int i = 0; i < it->second.rows(); ++i) {
      Real val = 0.0;
      Real objFuncPlus = 0.0;
      Real objFuncMinus = 0.0;
      val = this->targetEmbed.coeff(i, it->first); // Θ_i
      this->targetEmbed.coeffRef(i, it->first) = val + EPS;
      objFuncPlus = this->calcLoss(data, arg, true);
      this->targetEmbed.coeffRef(i, it->first) = val - EPS;
      objFuncMinus = this->calcLoss(data, arg, true);
      this->targetEmbed.coeffRef(i, it->first) = val;

      Real gradVal = it->second.coeff(i, 0);
      Real enumVal = (objFuncPlus - objFuncMinus)/ (2.0*EPS);
      if ((gradVal - enumVal) > 1.0e-05) {
	std::cout << "Grad: " << gradVal << std::endl;
	std::cout << "Enum: " << enumVal << std::endl;
      } else {}
    }
  }
}

void AttentionBiEncDec::makeTrans(const std::vector<int>& tgt, 
				  std::vector<int>& trans) {
  for (auto it = tgt.begin(); it != tgt.end(); ++it) {
    if (*it != this->targetVoc.eosIndex) {
      trans.push_back(*it);
    } else {}
  }
}

void AttentionBiEncDec::loadCorpus(const std::string& src, 
				   const std::string& tgt,
				   std::vector<AttentionBiEncDec::Data*>& data) {
  std::ifstream ifsSrc(src.c_str());
  std::ifstream ifsTgt(tgt.c_str());

  assert(ifsSrc);
  assert(ifsTgt);

  int numLine = 0;
  // Src
  for (std::string line; std::getline(ifsSrc, line);) {
    std::vector<std::string> tokens;
    AttentionBiEncDec::Data *dataTmp(NULL);
    dataTmp = new AttentionBiEncDec::Data;
    data.push_back(dataTmp);
    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data.back()->src.push_back(sourceVoc.tokenIndex.count(*it) ? sourceVoc.tokenIndex.at(*it) : sourceVoc.unkIndex);
    }
    data.back()->src.push_back(sourceVoc.eosIndex); // EOS
  }

  //Tgt
  for (std::string line; std::getline(ifsTgt, line);) {
    std::vector<std::string> tokens;

    Utils::split(line, tokens);

    for (auto it = tokens.begin(); it != tokens.end(); ++it) {
      data[numLine]->tgt.push_back(targetVoc.tokenIndex.count(*it) ? targetVoc.tokenIndex.at(*it) : targetVoc.unkIndex);
    }
    data[numLine]->tgt.push_back(targetVoc.eosIndex); // EOS
    ++numLine;
  }
}

std::tuple<std::string, std::string> AttentionBiEncDec::saveModel(AttentionBiEncDec::Grad& grad,
								  const float i) {
  std::ostringstream oss;
  oss << this->saveDirName << "Model_AttentionBiEncDec"
      << ".itr_" << i+1
      << ".BlackOut_" << (this->useBlackOut?"true":"false")
      << ".beamSize_" << this->beamSize 
      << ".miniBatchSize_" << this->miniBatchSize
      << ".threadNum_" << this->threadNum
      << ".lrSGD_"<< this->learningRate 
      << ".bin"; 

  this->save(oss.str());

  std::ostringstream ossGrad;
  ossGrad << this->saveDirName << "Model_AttentionBiEncDecGrad"
	  << ".itr_" << i+1
	  << ".BlackOut_" << (this->useBlackOut?"true":"false")
	  << ".beamSize_" << this->beamSize 
	  << ".miniBatchSize_" << this->miniBatchSize
	  << ".threadNum_" << this->threadNum
	  << ".lrSGD_"<< this->learningRate 
	  << ".bin"; 

  return std::forward_as_tuple(oss.str(), ossGrad.str());
}

void AttentionBiEncDec::loadModel(AttentionBiEncDec::Grad& grad, 
				  const std::string& loadModelName, 
				  const std::string& loadGradName) {
  this->load(loadModelName.c_str());
}

void AttentionBiEncDec::saveResult(const Real value, 
				   const std::string& name) {
  /* For Model Analysis */
  std::ofstream valueFile;
  std::ostringstream ossValue;
  ossValue << this->saveDirName << "Model_AttentionBiEncDec" << name;

  valueFile.open(ossValue.str(), std::ios::app); // open a file with 'a' mode

  valueFile << value << std::endl;
}

void AttentionBiEncDec::demo(const std::string& srcTrain, 
			     const std::string& tgtTrain, 
			     const std::string& srcDev, 
			     const std::string& tgtDev,
			     const int inputDim, 
			     const int hiddenEncDim,
			     const int hiddenDim,
			     const Real scale, 
			     const bool useBlackOut, 
			     const int blackOutSampleNum, 
			     const Real blackOutAlpha, 
			     const Real clipThreshold,
			     const int beamSize, 
			     const int maxLen,
			     const int miniBatchSize, 
			     const int threadNum,
			     const Real learningRate,
			     const int srcVocaThreshold, 
			     const int tgtVocaThreshold,
			     const std::string& saveDirName) {

  Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold); 

  std::vector<AttentionBiEncDec::Data*> trainData;
  std::vector<AttentionBiEncDec::Data*> devData;

  AttentionBiEncDec attentionBiEncDec(sourceVoc, 
				      targetVoc, 
				      trainData, 
				      devData, 
				      inputDim, 
				      hiddenEncDim, 
				      hiddenDim, 
				      scale,
				      useBlackOut, 
				      blackOutSampleNum, 
				      blackOutAlpha,
				      AttentionBiEncDec::SGD, 
				      clipThreshold,
				      beamSize, 
				      maxLen, 
				      miniBatchSize, 
				      threadNum,
				      learningRate, 
				      false,
				      0, 
				      saveDirName);

  attentionBiEncDec.loadCorpus(srcTrain, tgtTrain, trainData);
  attentionBiEncDec.loadCorpus(srcDev, tgtDev, devData); 

  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

  AttentionBiEncDec::Grad grad(attentionBiEncDec);
  // Test
  auto test = trainData[0];
  for (int i = 0; i < 100; ++i) {
    std::cout << "\nEpoch " << i+1 
	      << " (lr = " << attentionBiEncDec.learningRate << ")" << std::endl;

    bool status = attentionBiEncDec.trainOpenMP(grad); 

    if (!status){
      attentionBiEncDec.load(attentionBiEncDec.prevModelFileName);
      attentionBiEncDec.learningRate *= 0.5;
      --i;
      continue;
    }

    // Save a model
    attentionBiEncDec.saveModel(grad, i);
  
    std::vector<AttentionBiEncDec::ThreadArg> args;
    std::vector<std::vector<int> > translation(2);
    args.push_back(AttentionBiEncDec::ThreadArg(attentionBiEncDec));
    args.push_back(AttentionBiEncDec::ThreadArg(attentionBiEncDec));
    args[0].initTrans(1, attentionBiEncDec.maxLen);
    args[1].initTrans(5, attentionBiEncDec.maxLen);

    std::cout << "** Greedy Search" << std::endl;
    attentionBiEncDec.translate(test, args[0], translation[0], true);
    std::cout << "** Beam Search" << std::endl;
    attentionBiEncDec.translate(test, args[1], translation[1], true);
  }
}

void AttentionBiEncDec::demo(const std::string& srcTrain, 
			     const std::string& tgtTrain,
			     const std::string& srcDev, 
			     const std::string& tgtDev,
			     const int inputDim, 
			     const int hiddenEncDim, 
			     const int hiddenDim, 
			     const Real scale, 
			     const bool useBlackOut, 
			     const int blackOutSampleNum, 
			     const Real blackOutAlpha, 
			     const Real clipThreshold,
			     const int beamSize, 
			     const int maxLen,
			     const int miniBatchSize, 
			     const int threadNum,
			     const Real learningRate,
			     const int srcVocaThreshold, 
			     const int tgtVocaThreshold,
			     const std::string& saveDirName, 
			     const std::string& loadModelName, 
			     const std::string& loadGradName,
			     const int startIter) {
  Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold); 

  std::vector<AttentionBiEncDec::Data*> trainData, devData;

  AttentionBiEncDec attentionBiEncDec(sourceVoc, 
				      targetVoc, 
				      trainData, 
				      devData, 
				      inputDim, 
				      hiddenEncDim, 
				      hiddenDim, 
				      scale,
				      useBlackOut, 
				      blackOutSampleNum, 
				      blackOutAlpha,
				      AttentionBiEncDec::SGD, 
				      clipThreshold,
				      beamSize, 
				      maxLen, 
				      miniBatchSize, 
				      threadNum,
				      learningRate, 
				      false,
				      startIter, 
				      saveDirName);
  
  attentionBiEncDec.loadCorpus(srcTrain, tgtTrain, trainData);
  attentionBiEncDec.loadCorpus(srcDev, tgtDev, devData); 

  std::vector<AttentionBiEncDec::ThreadArg> args; // Evaluation of Dev.
  std::vector<std::vector<int> > translation(attentionBiEncDec.devData.size());
  for (int i = 0; i < threadNum; ++i){
    args.push_back(AttentionBiEncDec::ThreadArg(attentionBiEncDec));
    args.back().initTrans(1, 100); // Sentences consists of less than 50 words
  }

  std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
  std::cout << "# of Development Data:\t" << devData.size() << std::endl;
  std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
  std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;

  AttentionBiEncDec::Grad grad(attentionBiEncDec);
  // Model Loaded...
  attentionBiEncDec.loadModel(grad, loadModelName, loadGradName);
  attentionBiEncDec.prevModelFileName = loadModelName;
  attentionBiEncDec.prevGradFileName = loadGradName;

  // Test
  auto test = trainData[0];

  Real lossDev = 0.;
  Real tgtNum = 0.;

#pragma omp parallel for num_threads(attentionBiEncDec.threadNum) schedule(dynamic) // ThreadNum
  for (int i = 0; i < (int)devData.size(); ++i) {
    int id = omp_get_thread_num();
    Real loss = attentionBiEncDec.calcLoss(devData[i], args[id], false);
#pragma omp critical
    {
      lossDev += loss;
      tgtNum += devData[i]->tgt.size();
    }
  }

  Real currentDevPerp = exp(lossDev/tgtNum);
  std::cout << "Development Perplexity and Loss (/sentence):  " 
	    << currentDevPerp << ", "
	    << lossDev/devData.size() << "; "
	    << devData.size() << std::endl;
  attentionBiEncDec.prevPerp = currentDevPerp;

  for (int i = 0; i < startIter; ++i) {
   attentionBiEncDec.rnd.shuffle(attentionBiEncDec.trainData);
  }
  for (int i = startIter; i < 100; ++i) {
    std::cout << "\nEpoch " << i+1 
	      << " (lr = " << attentionBiEncDec.learningRate 
	      << ")" << std::endl;
    
    bool status = attentionBiEncDec.trainOpenMP(grad); 
    if (!status){
      attentionBiEncDec.loadModel(grad, attentionBiEncDec.prevModelFileName, attentionBiEncDec.prevGradFileName);
      attentionBiEncDec.learningRate *= 0.5;
      --i;
      continue;
    }

    // Save a model
    attentionBiEncDec.saveModel(grad, i);

    std::vector<AttentionBiEncDec::ThreadArg> argsTmp;
    std::vector<std::vector<int> > translation(2);
    argsTmp.push_back(AttentionBiEncDec::ThreadArg(attentionBiEncDec));
    argsTmp.push_back(AttentionBiEncDec::ThreadArg(attentionBiEncDec));
    argsTmp[0].initTrans(1, attentionBiEncDec.maxLen);
    argsTmp[1].initTrans(5, attentionBiEncDec.maxLen);

    std::cout << "** Greedy Search" << std::endl;
    attentionBiEncDec.translate(test, argsTmp[0], translation[0], true);
    std::cout << "** Beam Search" << std::endl;
    attentionBiEncDec.translate(test, argsTmp[1], translation[1], true);
  }
}

void AttentionBiEncDec::evaluate(const std::string& srcTrain, 
				 const std::string& tgtTrain,
				 const std::string& srcTest, 
				 const std::string& tgtTest,
				 const int inputDim, 
				 const int hiddenEncDim, 
				 const int hiddenDim, 
				 const Real scale,
				 const bool useBlackOut, 
				 const int blackOutSampleNum, 
				 const Real blackOutAlpha, 
				 const int beamSize, 
				 const int maxLen,
				 const int miniBatchSize, 
				 const int threadNum,
				 const Real learningRate,
				 const int srcVocaThreshold, 
				 const int tgtVocaThreshold,
				 const bool isTest, 
				 const std::string& saveDirName, 
				 const std::string& loadModelName, 
				 const std::string& loadGradName, 
				 const int startIter) {
  static Vocabulary sourceVoc(srcTrain, srcVocaThreshold);
  static Vocabulary targetVoc(tgtTrain, tgtVocaThreshold); 
  static std::vector<AttentionBiEncDec::Data*> trainData, testData;

  static AttentionBiEncDec attentionBiEncDec(sourceVoc, 
					     targetVoc, 
					     trainData, 
					     testData, 
					     inputDim, 
					     hiddenEncDim, 
					     hiddenDim, 
					     scale,
					     useBlackOut, 
					     blackOutSampleNum, 
					     blackOutAlpha, 
					     AttentionBiEncDec::SGD, 
					     3.0,
					     beamSize, 
					     maxLen, 
					     miniBatchSize, 
					     threadNum,
					     learningRate, 
					     true, 
					     startIter, 
					     saveDirName);
  
  if (testData.empty()) {
      attentionBiEncDec.loadCorpus(srcTest, tgtTest, testData); 
      std::cout << "# of Training Data:\t" << trainData.size() << std::endl;
      std::cout << "# of Evaluation Data:\t" << testData.size() << std::endl;
      std::cout << "Source voc size: " << sourceVoc.tokenIndex.size() << std::endl;
      std::cout << "Target voc size: " << targetVoc.tokenIndex.size() << std::endl;
  } else {}
  std::vector<AttentionBiEncDec::ThreadArg> args; // Evaluation of Test
  std::vector<std::vector<int> > translation(testData.size());
  for (int i = 0; i < threadNum; ++i) {
    args.push_back(AttentionBiEncDec::ThreadArg(attentionBiEncDec));
    args.back().initTrans(attentionBiEncDec.beamSize, attentionBiEncDec.maxLen);
  }

  AttentionBiEncDec::Grad grad(attentionBiEncDec);
  // Model Loaded...
  attentionBiEncDec.loadModel(grad, loadModelName, loadGradName);

  Real lossTest = 0.;
  Real tgtNum = 0.;

#pragma omp parallel for num_threads(attentionBiEncDec.threadNum) schedule(dynamic) shared (args) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    Real loss;
    int id = omp_get_thread_num();
    loss = attentionBiEncDec.calcLoss(testData[i], args[id], false);
#pragma omp critical
    {
      lossTest += loss;
      tgtNum += testData[i]->tgt.size(); // include `*EOS*`
    }
  }

  std::cout << "Perplexity and Loss (/sentence):  " 
	    << exp(lossTest/tgtNum) << ", "
	    << lossTest/testData.size() << "; "
	    << testData.size() << std::endl;

#pragma omp parallel for num_threads(attentionBiEncDec.threadNum) schedule(dynamic) // ThreadNum
  for (int i = 0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    int id = omp_get_thread_num();
    attentionBiEncDec.translate(evalData, args[id], translation[i], false);
  }

  std::ofstream outputFile;
  std::ostringstream oss;
  std::string parsedMode;
  oss << attentionBiEncDec.saveDirName << "Model_AttentionBiEncDec"
      << ".BlackOut_" << (attentionBiEncDec.useBlackOut?"true":"false")
      << ".beamSize_" << attentionBiEncDec.beamSize 
      << ".lrSGD_" << attentionBiEncDec.learningRate 
      << ".startIter_" << startIter
      << ".Output" << (attentionBiEncDec.isTest?"Test":"Dev")
      << ".translate";
  outputFile.open(oss.str(), std::ios::out);

  for (int i = 0; i < (int)testData.size(); ++i) {
    auto evalData = testData[i];
    for (auto it = evalData->trans.begin(); it != evalData->trans.end(); ++it) {
      outputFile << attentionBiEncDec.targetVoc.tokenList[*it]->str << " ";
    }
    outputFile << std::endl;
    // trans
    testData[i]->trans.clear();

  }
}

void AttentionBiEncDec::save(const std::string& fileName) {
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);  
  
  this->enc.save(ofs);
  this->encRev.save(ofs);
  this->dec.save(ofs);

  this->decInitAffine.save(ofs);
  this->stildeAffine.save(ofs);

  this->softmax.save(ofs);
  this->blackOut.save(ofs);

  Utils::save(ofs, sourceEmbed);
  Utils::save(ofs, targetEmbed);

  Utils::save(ofs, Wgeneral);
}

void AttentionBiEncDec::load(const std::string& fileName) {
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  this->enc.load(ifs);
  this->encRev.load(ifs);
  this->dec.load(ifs);

  this->decInitAffine.load(ifs);
  this->stildeAffine.load(ifs);

  this->softmax.load(ifs);
  this->blackOut.load(ifs);

  Utils::load(ifs, sourceEmbed);
  Utils::load(ifs, targetEmbed);
  
  Utils::load(ifs, Wgeneral);
}

/* AttentionBiEncDec::DecCandidate */
void AttentionBiEncDec::DecCandidate::init(const int maxLen) {
  this->score = 0.0;
  this->generatedTarget.clear();
  this->stop = false;

  if (this->decState.empty()) {
    for (int i = 0; i < maxLen; ++i) {
      LSTM::State *lstmDecState(NULL);
      lstmDecState = new LSTM::State;
      this->decState.push_back(lstmDecState);
    }
  }
}

/* AttentionBiEncDec::ThreadFunc */
AttentionBiEncDec::ThreadArg::ThreadArg(AttentionBiEncDec& attentionBiEncDec) {
  // LSTM
  int stildeSize = attentionBiEncDec.hiddenDim + attentionBiEncDec.hiddenEncDim*2;
  for (int i = 0; i < 150; ++i) {
    LSTM::State *lstmState(NULL);
    lstmState = new LSTM::State;
    this->encState.push_back(lstmState);
    LSTM::State *lstmStateRev(NULL);
    lstmStateRev = new LSTM::State;
    this->encRevState.push_back(lstmStateRev);
    this->biEncState.push_back(VecD(attentionBiEncDec.hiddenEncDim*2));

    // Vectors or Matrices
    this->s_tilde.push_back(VecD(attentionBiEncDec.hiddenDim));
    this->contextSeqList.push_back(VecD(attentionBiEncDec.hiddenEncDim*2));
    this->del_stilde.push_back(VecD(attentionBiEncDec.hiddenDim));
    // Affine
    this->stildeEnd.push_back(VecD(stildeSize));
  }

  for (int i = 0; i < attentionBiEncDec.maxLen; ++i) {
    LSTM::State *lstmDecState(NULL);
    lstmDecState = new LSTM::State;
    this->decState.push_back(lstmDecState);
  }

  if (attentionBiEncDec.useBlackOut){
    for (int i = 0; i <attentionBiEncDec.maxLen; ++i) {
      this->blackOutState.push_back(BlackOut::State(attentionBiEncDec.blackOut));
      this->targetDistVec.push_back(VecD());
    }
  }

  // Vectors or Matrices
  this->decInitStateEnd = VecD(attentionBiEncDec.hiddenEncDim*2);
}

void AttentionBiEncDec::ThreadArg::initTrans(const int beamSize, 
					     const int maxLen) {
  for (int i = 0; i < beamSize; ++i) {
    this->candidate.push_back(AttentionBiEncDec::DecCandidate());
    this->candidate.back().init(maxLen);
  }
}

void AttentionBiEncDec::ThreadArg::initLoss() {
  this->loss = 0.0;
}

void AttentionBiEncDec::ThreadArg::init(AttentionBiEncDec& attentionBiEncDec, 
					const AttentionBiEncDec::Data* data, 
					const bool train) {
  this->srcLen = data->src.size();
  this->tgtLen = data->tgt.size();

  if (train) {
    this->alphaSeq = MatD::Zero(this->srcLen, this->tgtLen);
    this->decInitStateEnd = attentionBiEncDec.zeros2;
    this->del_decInitStateEnd = attentionBiEncDec.zeros2;
    this->del_alphaSeq = VecD(this->srcLen);
    this->del_alphaSeqTmp = attentionBiEncDec.zeros;
    this->del_WgeneralTmp = attentionBiEncDec.zeros2;
    this->alphaSeqVec = VecD(this->srcLen);

  } else {
    this->alphaSeqVec = VecD::Zero(this->srcLen);
  }
}

/* AttentionBiEncDec::Grad */
AttentionBiEncDec::Grad::Grad(AttentionBiEncDec& attentionBiEncDec):
  gradHist(0)
{
  this->lstmSrcGrad = LSTM::Grad(attentionBiEncDec.enc);
  this->lstmSrcRevGrad = LSTM::Grad(attentionBiEncDec.encRev);
  this->lstmTgtGrad = LSTM::Grad(attentionBiEncDec.dec);

  this->decInitAffineGrad = Affine::Grad(attentionBiEncDec.decInitAffine);
  this->stildeAffineGrad = Affine::Grad(attentionBiEncDec.stildeAffine);

  if (!attentionBiEncDec.useBlackOut) {
    this->softmaxGrad = SoftMax::Grad(attentionBiEncDec.softmax);
  } else {
    this->blackOutGrad = BlackOut::Grad(attentionBiEncDec.blackOut, false);
  }

  this->Wgeneral = MatD::Zero(attentionBiEncDec.Wgeneral.rows(), attentionBiEncDec.Wgeneral.cols());

  this->init();
}

void AttentionBiEncDec::Grad::init() {
    this->sourceEmbed.clear();
    this->targetEmbed.clear();

    this->lstmSrcGrad.init();
    this->lstmSrcRevGrad.init();
    this->lstmTgtGrad.init();

    this->decInitAffineGrad.init();
    this->stildeAffineGrad.init();

    this->softmaxGrad.init();
    this->blackOutGrad.init();

    this->Wgeneral.setZero();
}

Real AttentionBiEncDec::Grad::norm() {
  Real res = 0.0; 

  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }
  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it){
    res += it->second.squaredNorm();
  }

  res += this->lstmSrcGrad.norm();
  res += this->lstmSrcRevGrad.norm();
  res += this->lstmTgtGrad.norm();

  res += this->decInitAffineGrad.norm();
  res += this->stildeAffineGrad.norm();

  res += this->softmaxGrad.norm();
  res += this->blackOutGrad.norm();

  res += this->Wgeneral.squaredNorm();

  return res;
}

void AttentionBiEncDec::Grad::operator += (const AttentionBiEncDec::Grad& grad) {
  for (auto it = grad.sourceEmbed.begin(); it != grad.sourceEmbed.end(); ++it){
    if (this->sourceEmbed.count(it->first)){
      this->sourceEmbed.at(it->first) += it->second;
    } else {
      this->sourceEmbed[it->first] = it->second;
    }
  }

  for (auto it = grad.targetEmbed.begin(); it != grad.targetEmbed.end(); ++it){
    if (this->targetEmbed.count(it->first)){
      this->targetEmbed.at(it->first) += it->second;
    } else {
      this->targetEmbed[it->first] = it->second;
    }
  }

  this->lstmSrcGrad += grad.lstmSrcGrad;
  this->lstmSrcRevGrad += grad.lstmSrcRevGrad;
  this->lstmTgtGrad += grad.lstmTgtGrad;

  this->decInitAffineGrad += grad.decInitAffineGrad;
  this->stildeAffineGrad += grad.stildeAffineGrad;

  this->softmaxGrad += grad.softmaxGrad;  
  this->blackOutGrad += grad.blackOutGrad;  

  this->Wgeneral += grad.Wgeneral;
}

void AttentionBiEncDec::Grad::sgd(AttentionBiEncDec& attentionBiEncDec, 
				  const Real learningRate) {
  for (auto it = this->sourceEmbed.begin(); it != this->sourceEmbed.end(); ++it) {
    attentionBiEncDec.sourceEmbed.col(it->first) -= learningRate * it->second;
  }
  for (auto it = this->targetEmbed.begin(); it != this->targetEmbed.end(); ++it) {
    attentionBiEncDec.targetEmbed.col(it->first) -= learningRate * it->second;
  }

  this->lstmSrcGrad.sgd(learningRate, attentionBiEncDec.enc);
  this->lstmSrcRevGrad.sgd(learningRate, attentionBiEncDec.encRev);
  this->lstmTgtGrad.sgd(learningRate, attentionBiEncDec.dec);

  this->decInitAffineGrad.sgd(learningRate, attentionBiEncDec.decInitAffine);
  this->stildeAffineGrad.sgd(learningRate, attentionBiEncDec.stildeAffine);

  if (!attentionBiEncDec.useBlackOut) {
    this->softmaxGrad.sgd(learningRate, attentionBiEncDec.softmax);
  } else {
    this->blackOutGrad.sgd(learningRate, attentionBiEncDec.blackOut);
  }

  Optimizer::sgd(this->Wgeneral, learningRate, attentionBiEncDec.Wgeneral);
}

void AttentionBiEncDec::Grad::save(AttentionBiEncDec& attentionBiEncDec, 
				   const std::string& filename) {
  std::ofstream ofs(filename.c_str(), std::ios::out|std::ios::binary);
  assert(ofs);

  Utils::save(ofs, this->gradHist->sourceEmbedMatGrad);
  Utils::save(ofs, this->gradHist->targetEmbedMatGrad);

  // LSTM
  this->lstmSrcGrad.saveHist(ofs);
  this->lstmSrcRevGrad.saveHist(ofs);
  this->lstmTgtGrad.saveHist(ofs);

  // Affine
  this->decInitAffineGrad.saveHist(ofs);
  this->stildeAffineGrad.saveHist(ofs);

  if (attentionBiEncDec.useBlackOut) {
    this->blackOutGrad.saveHist(ofs);
  } else {
    this->softmaxGrad.saveHist(ofs);
  }

  Utils::save(ofs, this->gradHist->WgeneralMatGrad);
}

void AttentionBiEncDec::Grad::load(AttentionBiEncDec& attentionBiEncDec, 
				   const std::string& filename) {
  std::ifstream ifs(filename.c_str(), std::ios::in|std::ios::binary);
  assert(ifs);

  Utils::load(ifs, this->gradHist->sourceEmbedMatGrad);
  Utils::load(ifs, this->gradHist->targetEmbedMatGrad);

  // LSTM
  this->lstmSrcGrad.loadHist(ifs);
  this->lstmSrcRevGrad.loadHist(ifs);
  this->lstmTgtGrad.loadHist(ifs);

  // Affine
  this->decInitAffineGrad.loadHist(ifs);
  this->stildeAffineGrad.loadHist(ifs);

  if (attentionBiEncDec.useBlackOut) {
    this->blackOutGrad.loadHist(ifs);
  } else {
    this->softmaxGrad.loadHist(ifs);
  }
  
  Utils::load(ifs, this->gradHist->WgeneralMatGrad);
}
