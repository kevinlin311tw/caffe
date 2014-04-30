// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}

template <typename Dtype>
Dtype MultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    loss -= log(prob);
  }
  return loss / num;
}

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    bottom_diff[i * dim + label] = -1. / prob / num;
  }
}


template <typename Dtype>
void InfogainLossLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(this->layer_param_.infogain_loss_param().source(),
                          &blob_proto);
  infogain_.FromProto(blob_proto);
  CHECK_EQ(infogain_.num(), 1);
  CHECK_EQ(infogain_.channels(), 1);
  CHECK_EQ(infogain_.height(), infogain_.width());
}


template <typename Dtype>
Dtype InfogainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* infogain_mat = infogain_.cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  CHECK_EQ(infogain_.height(), dim);
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      loss -= infogain_mat[label * dim + j] * log(prob);
    }
  }
  return loss / num;
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* bottom_label = (*bottom)[1]->cpu_data();
  const Dtype* infogain_mat = infogain_.cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  int num = (*bottom)[0]->num();
  int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
  CHECK_EQ(infogain_.height(), dim);
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    for (int j = 0; j < dim; ++j) {
      Dtype prob = max(bottom_data[i * dim + j], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + j] = - infogain_mat[label * dim + j] / prob / num;
    }
  }
}


template <typename Dtype>
void EuclideanLossLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and labels should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  difference_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  int num = bottom[0]->num();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
      difference_.mutable_cpu_data());
  Dtype loss = caffe_cpu_dot(
      count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const int count = (*bottom)[0]->count();
  const int num = (*bottom)[0]->num();
  for (int bottom_id = 0; bottom_id < 2; ++bottom_id) {
    if (propagate_down[bottom_id]) {
      // Compute the gradient
      const int sign = bottom_id ? -1 : 1;
      caffe_cpu_axpby(count, Dtype(sign) / num, difference_.cpu_data(),
          Dtype(0), (*bottom)[bottom_id]->mutable_cpu_diff());
    }
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 2, 1, 1);
}

template <typename Dtype>
Dtype AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  Dtype logprob = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      if (bottom_data[i * dim + j] > maxval) {
        maxval = bottom_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == static_cast<int>(bottom_label[i])) {
      ++accuracy;
    }
    Dtype prob = max(bottom_data[i * dim + static_cast<int>(bottom_label[i])],
                     Dtype(kLOG_THRESHOLD));
    logprob -= log(prob);
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  (*top)[0]->mutable_cpu_data()[1] = logprob / num;
  // Accuracy layer should not be used as a loss function.
  return Dtype(0);
}

template <typename Dtype>
void HingeLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  dim_ = count_ / num_;
  CHECK_EQ(num_, bottom[1]->num())
      << "The data and labels should have the same number.";
  loss_per_datum_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
Dtype HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* loss_per_datum = loss_per_datum_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  for (int i = 0; i < num_; ++i) {
    const int label_index = static_cast<int>(label[i]);
    for (int j = 0; j < dim_; ++j) {
      const int sign = (j == label_index) ? -1 : 1;
      loss_per_datum[i * dim_ + j] =
          max(Dtype(0), 1 + sign * bottom_data[i * dim_ + j]);
    }
  }
  return caffe_cpu_asum(count_, loss_per_datum) / num_;
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    NOT_IMPLEMENTED;  // Cannot backprop to labels.
  }
  if (propagate_down[0]) {
    const Dtype* loss_per_datum = loss_per_datum_.cpu_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* label = (*bottom)[1]->cpu_data();
    caffe_cpu_sign(count_, loss_per_datum, bottom_diff);
    for (int i = 0; i < num_; ++i) {
      bottom_diff[i * dim_ + static_cast<int>(label[i])] *= -1;
    }
    caffe_scal(count_, Dtype(1. / num_), bottom_diff);
  }
}

INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
INSTANTIATE_CLASS(InfogainLossLayer);
INSTANTIATE_CLASS(EuclideanLossLayer);
INSTANTIATE_CLASS(AccuracyLayer);
INSTANTIATE_CLASS(HingeLossLayer);

}  // namespace caffe
