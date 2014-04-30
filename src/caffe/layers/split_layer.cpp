// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top->size(); ++i) {
    if (bottom[0] != (*top)[i]) {
      (*top)[i]->ReshapeLike(*bottom[0]);
    }
  }
}

template <typename Dtype>
Dtype SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < top->size(); ++i) {
    if (bottom[0] != (*top)[i]) {
      caffe_copy(count_, bottom[0]->cpu_data(), (*top)[i]->mutable_cpu_data());
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    // Add remaining top blob diffs.
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    if (top_diff != bottom_diff) {
      caffe_copy(count_, top_diff, bottom_diff);
    }
    for (int i = 1; i < top.size(); ++i) {
      top_diff = top[i]->cpu_diff();
      caffe_add(count_, top_diff, bottom_diff, bottom_diff);
    }
  }
}


INSTANTIATE_CLASS(SplitLayer);

}  // namespace caffe
