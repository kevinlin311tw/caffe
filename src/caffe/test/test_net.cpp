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

namespace caffe {


template <typename Dtype>
class NetTest : public ::testing::Test {
 protected:
  virtual void SetUp() {  // Create the leveldb
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

  virtual void InitNetFromProto() {
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(*proto_, &param));
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
    proto_.reset(new string(proto_prefix + "source: '" + *filename_ +
                            "' " + proto_suffix));
    InitNetFromProto();
  }

  virtual void InitTrickyNet() {
    const string& proto_prefix =
        "name: 'TrickyTestNetwork' "
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
    proto_.reset(new string(proto_prefix + "source: '" + *filename_ +
                            "' " + proto_suffix));
    InitNetFromProto();
  }

  shared_ptr<string> filename_;
  shared_ptr<string> proto_;
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

// TYPED_TEST(NetTest, TestBlobNumConsumersTinyNet) {
//   this->InitTinyNet();
//   const map<string, int>& blob_names_index = this->net_->blob_names_index();
//   const vector<int>& blob_num_consumers = this->net_->blob_num_consumers();
//   EXPECT_EQ(1, blob_num_consumers[blob_names_index.find("data")->second]);
//   EXPECT_EQ(1, blob_num_consumers[blob_names_index.find("label")->second]);
//   EXPECT_EQ(1,
//        blob_num_consumers[blob_names_index.find("innerproduct")->second]);
// }
// 
// TYPED_TEST(NetTest, TestBlobNumConsumersTrickyNet) {
//   this->InitTrickyNet();
//   const map<string, int>& blob_names_index = this->net_->blob_names_index();
//   const vector<int>& blob_num_consumers = this->net_->blob_num_consumers();
//   EXPECT_EQ(5, blob_num_consumers[blob_names_index.find("data")->second]);
//   EXPECT_EQ(2, blob_num_consumers[blob_names_index.find("label")->second]);
//   EXPECT_EQ(1,
//       blob_num_consumers[blob_names_index.find("eltwiseproduct")->second]);
//   EXPECT_EQ(0,
//       blob_num_consumers[blob_names_index.find("accuracy")->second]);
// }

}  // namespace caffe
