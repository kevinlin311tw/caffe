// Copyright 2014 BVLC and contributors.

#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::make_pair;

namespace caffe {


template <typename Dtype>
class NetTest : public ::testing::Test {
 protected:
  virtual void SetUp() {  // Create the leveldb
    seed_ = 1701;
    Caffe::set_random_seed(seed_);
    filename_.reset(new string(tmpnam(NULL)));  // get temp name
    LOG(INFO) << "Using temporary leveldb " << *filename_;
    leveldb::DB* db;
    leveldb::Options options;
    options.error_if_exists = true;
    options.create_if_missing = true;
    leveldb::Status status =
        leveldb::DB::Open(options, filename_->c_str(), &db);
    CHECK(status.ok());
    for (int i = 0; i < 5; ++i) {
      Datum datum;
      datum.set_label(i);
      datum.set_channels(2);
      datum.set_height(3);
      datum.set_width(4);
      std::string* data = datum.mutable_data();
      for (int j = 0; j < 24; ++j) {
        data->push_back(static_cast<uint8_t>(i));
      }
      std::stringstream ss;
      ss << i;
      db->Put(leveldb::WriteOptions(), ss.str(), datum.SerializeAsString());
    }
    delete db;
  }

  virtual void InitNetFromProto(const string& proto) {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    net_.reset(new Net<Dtype>(param));
  }

  virtual void InitTinyNet() {
    const string& proto_prefix =
        "name: 'TinyTestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DATA "
        "  data_param { ";
    const string& proto_suffix =
        "    batch_size: 1 "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'innerproduct' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 1000 "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "    bias_filler { "
        "      type: 'constant' "
        "      value: 0 "
        "    } "
        "  } "
        "  blobs_lr: 1. "
        "  blobs_lr: 2. "
        "  weight_decay: 1. "
        "  weight_decay: 0. "
        "  bottom: 'data' "
        "  top: 'innerproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'innerproduct' "
        "  bottom: 'label' "
        "} ";
    InitNetFromProto(proto_prefix + "source: '" + *filename_ +
                     "' " + proto_suffix);
  }

  virtual void InitTrickyNet() {
    const string& proto =
        "name: 'TrickyTestNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    num: 5 "
        "    channels: 1 "
        "    height: 1 "
        "    width: 1 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "  top: 'label' "
        "} "
        "layers: { "
        "  name: 'eltproduct' "
        "  type: ELTWISE_PRODUCT "
        "  bottom: 'data' "
        "  bottom: 'data' "
        "  bottom: 'data' "
        "  bottom: 'data' "
        "  top: 'eltwiseproduct' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: SOFTMAX_LOSS "
        "  bottom: 'eltwiseproduct' "
        "  bottom: 'label' "
        "} "
        "layers: { "
        "  name: 'accuracy' "
        "  type: ACCURACY "
        "  bottom: 'data' "
        "  bottom: 'label' "
        "  top: 'accuracy' "
        "} ";
    InitNetFromProto(proto);
  }

  virtual void InitUnsharedWeightsNet() {
    const string& proto =
        "name: 'UnsharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  blob_name: 'unsharedweights1' "
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  blob_name: 'unsharedweights2' "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProto(proto);
  }

  virtual void InitSharedWeightsNet() {
    const string& proto =
        "name: 'SharedWeightsNetwork' "
        "layers: { "
        "  name: 'data' "
        "  type: DUMMY_DATA "
        "  dummy_data_param { "
        "    num: 5 "
        "    channels: 2 "
        "    height: 3 "
        "    width: 4 "
        "    data_filler { "
        "      type: 'gaussian' "
        "      std: 0.01 "
        "    } "
        "  } "
        "  top: 'data' "
        "} "
        "layers: { "
        "  name: 'innerproduct1' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  blob_name: 'sharedweights' "
        "  bottom: 'data' "
        "  top: 'innerproduct1' "
        "} "
        "layers: { "
        "  name: 'innerproduct2' "
        "  type: INNER_PRODUCT "
        "  inner_product_param { "
        "    num_output: 10 "
        "    bias_term: false "
        "    weight_filler { "
        "      type: 'gaussian' "
        "      std: 10 "
        "    } "
        "  } "
        "  blob_name: 'sharedweights' "
        "  bottom: 'data' "
        "  top: 'innerproduct2' "
        "} "
        "layers: { "
        "  name: 'loss' "
        "  type: EUCLIDEAN_LOSS "
        "  bottom: 'innerproduct1' "
        "  bottom: 'innerproduct2' "
        "} ";
    InitNetFromProto(proto);
  }

  int seed_;
  shared_ptr<string> filename_;
  shared_ptr<Net<Dtype> > net_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(NetTest, Dtypes);

TYPED_TEST(NetTest, TestHasBlob) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_blob("data"));
  EXPECT_TRUE(this->net_->has_blob("label"));
  EXPECT_TRUE(this->net_->has_blob("innerproduct"));
  EXPECT_FALSE(this->net_->has_blob("loss"));
}

TYPED_TEST(NetTest, TestGetBlob) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->blob_by_name("data"), this->net_->blobs()[0]);
  EXPECT_EQ(this->net_->blob_by_name("label"), this->net_->blobs()[1]);
  EXPECT_EQ(this->net_->blob_by_name("innerproduct"), this->net_->blobs()[2]);
  EXPECT_FALSE(this->net_->blob_by_name("loss"));
}

TYPED_TEST(NetTest, TestHasLayer) {
  this->InitTinyNet();
  EXPECT_TRUE(this->net_->has_layer("data"));
  EXPECT_TRUE(this->net_->has_layer("innerproduct"));
  EXPECT_TRUE(this->net_->has_layer("loss"));
  EXPECT_FALSE(this->net_->has_layer("label"));
}

TYPED_TEST(NetTest, TestGetLayerByName) {
  this->InitTinyNet();
  EXPECT_EQ(this->net_->layer_by_name("data"), this->net_->layers()[0]);
  EXPECT_EQ(this->net_->layer_by_name("innerproduct"), this->net_->layers()[1]);
  EXPECT_EQ(this->net_->layer_by_name("loss"), this->net_->layers()[2]);
  EXPECT_FALSE(this->net_->layer_by_name("label"));
}

TYPED_TEST(NetTest, TestBlobBottomIndex) {
  this->InitTrickyNet();
  const map<string, int>& blob_names_index = this->net_->blob_names_index();
  const vector<vector<pair<int, int> > >& blob_bottom_indices =
      this->net_->blob_bottom_indices();
  EXPECT_EQ(4, blob_bottom_indices.size());
  const vector<pair<int, int> >& data_bottom_indices =
      blob_bottom_indices[blob_names_index.find("data")->second];
  EXPECT_EQ(5, data_bottom_indices.size());
  EXPECT_EQ(make_pair(1, 0), data_bottom_indices[0]);
  EXPECT_EQ(make_pair(1, 1), data_bottom_indices[1]);
  EXPECT_EQ(make_pair(1, 2), data_bottom_indices[2]);
  EXPECT_EQ(make_pair(1, 3), data_bottom_indices[3]);
  EXPECT_EQ(make_pair(3, 0), data_bottom_indices[4]);
  const vector<pair<int, int> >& label_bottom_indices =
      blob_bottom_indices[blob_names_index.find("label")->second];
  EXPECT_EQ(2, label_bottom_indices.size());
  EXPECT_EQ(make_pair(2, 1), label_bottom_indices[0]);
  EXPECT_EQ(make_pair(3, 1), label_bottom_indices[1]);
  const vector<pair<int, int> >& eltprod_bottom_indices =
      blob_bottom_indices[blob_names_index.find("eltwiseproduct")->second];
  EXPECT_EQ(1, eltprod_bottom_indices.size());
  EXPECT_EQ(make_pair(2, 0), eltprod_bottom_indices[0]);
  const vector<pair<int, int> >& accuracy_bottom_indices =
      blob_bottom_indices[blob_names_index.find("accuracy")->second];
  EXPECT_EQ(0, accuracy_bottom_indices.size());
}

TYPED_TEST(NetTest, TestBlobTopIndexTrickyNet) {
  this->InitTrickyNet();
  const map<string, int>& blob_names_index = this->net_->blob_names_index();
  const vector<pair<int, int> >& blob_top_index = this->net_->blob_top_index();
  const pair<int, int>& data_top_index =
      blob_top_index[blob_names_index.find("data")->second];
  EXPECT_EQ(make_pair(0, 0), data_top_index);
  const pair<int, int>& label_top_index =
      blob_top_index[blob_names_index.find("label")->second];
  EXPECT_EQ(make_pair(0, 1), label_top_index);
  const pair<int, int>& eltprod_top_index =
      blob_top_index[blob_names_index.find("eltwiseproduct")->second];
  EXPECT_EQ(make_pair(1, 0), eltprod_top_index);
  const pair<int, int>& accuracy_top_index =
      blob_top_index[blob_names_index.find("accuracy")->second];
  EXPECT_EQ(make_pair(3, 0), accuracy_top_index);
}

TYPED_TEST(NetTest, TestBlobBottomDiffScales) {
  this->InitTrickyNet();
  const vector<vector<bool> >& bottom_diff_scales =
      this->net_->bottom_diff_scales();
  EXPECT_EQ(4, bottom_diff_scales.size());
  EXPECT_EQ(0, bottom_diff_scales[0].size());
  EXPECT_EQ(4, bottom_diff_scales[1].size());
  EXPECT_EQ(2, bottom_diff_scales[2].size());
  EXPECT_EQ(2, bottom_diff_scales[3].size());
  EXPECT_EQ(0, bottom_diff_scales[1][0]);
  EXPECT_EQ(1, bottom_diff_scales[1][1]);
  EXPECT_EQ(1, bottom_diff_scales[1][2]);
  EXPECT_EQ(1, bottom_diff_scales[1][3]);
  EXPECT_EQ(0, bottom_diff_scales[2][0]);
  EXPECT_EQ(0, bottom_diff_scales[2][1]);
  EXPECT_EQ(1, bottom_diff_scales[3][0]);
  EXPECT_EQ(1, bottom_diff_scales[3][1]);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDataNet) {
  this->InitUnsharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_GT(loss, 0);
}

TYPED_TEST(NetTest, TestSharedWeightsDataNet) {
  this->InitSharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss;
  this->net_->Forward(bottom, &loss);
  EXPECT_FLOAT_EQ(loss, 0);
}

TYPED_TEST(NetTest, TestUnsharedWeightsDiffNet) {
  this->InitUnsharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss;
  this->net_->Forward(bottom, &loss);
  this->net_->Backward();
  EXPECT_GT(loss, 0);
  const int count = this->net_->layers()[1]->blobs()[0]->count();
  const TypeParam* grad1 = this->net_->layers()[1]->blobs()[0]->cpu_diff();
  const TypeParam* grad2 = this->net_->layers()[2]->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_GT(fabs(grad1[i]), 0);
    EXPECT_FLOAT_EQ(-1 * grad1[i], grad2[i]);
  }
}

TYPED_TEST(NetTest, TestSharedWeightsDiffNet) {
  this->InitSharedWeightsNet();
  vector<Blob<TypeParam>*> bottom;
  TypeParam loss;
  this->net_->Forward(bottom, &loss);
  this->net_->Backward();
  EXPECT_FLOAT_EQ(loss, 0);
  const int count = this->net_->layers()[1]->blobs()[0]->count();
  const TypeParam* grad1 = this->net_->layers()[1]->blobs()[0]->cpu_diff();
  const TypeParam* grad2 = this->net_->layers()[2]->blobs()[0]->cpu_diff();
  for (int i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(0, grad1[i]);
    EXPECT_FLOAT_EQ(0, grad2[i]);
  }
}

}  // namespace caffe
