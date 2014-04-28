// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DummyDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "DummyData Layer takes no input blobs.";
  const int num_top = top->size();
  CHECK_GE(num_top, 1)
      << "DummyData Layer produces at least one blob as output.";
  const DummyDataParameter& param = this->layer_param_.dummy_data_param();
  const int num_data_filler = param.data_filler_size();
  CHECK(num_data_filler == 0 || num_data_filler == 1 ||
        num_data_filler == num_top)
      << "Number of data fillers must be 0, 1 or equal to the number of tops; "
      << "was: " << num_data_filler;
  CHECK_EQ(num_top, param.num_size());
  CHECK_EQ(num_top, param.channels_size());
  CHECK_EQ(num_top, param.height_size());
  CHECK_EQ(num_top, param.width_size());
  fillers_.resize(num_top);
  FillerParameter filler_param;
  bool first_filler_set = false;
  if (num_data_filler == 0) {
    first_filler_set = true;
    filler_param.set_type("constant");
    filler_param.set_value(0);
  } else if (num_data_filler == 1) {
    first_filler_set = true;
    filler_param.CopyFrom(param.data_filler(0));
  }
  if (first_filler_set) {
    fillers_[0].reset(GetFiller<Dtype>(filler_param));
    for (int i = 1; i < num_top; ++i) {
      fillers_[i] = fillers_[0];
    }
  } else {
    for (int i = 0; i < num_top; ++i) {
      fillers_[i].reset(GetFiller<Dtype>(param.data_filler(i)));
    }
  }
  for (int i = 0; i < num_top; ++i) {
    (*top)[i]->Reshape(param.num(i), param.channels(i),
                       param.height(i), param.width(i));
  }
}

template <typename Dtype>
Dtype DummyDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < top->size(); ++i) {
    fillers_[i]->Fill((*top)[i]);
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(DummyDataLayer);

}  // namespace caffe
