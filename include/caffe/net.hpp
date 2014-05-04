// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::map;
using std::pair;
using std::set;
using std::string;
using std::vector;

namespace caffe {


template <typename Dtype>
class Net {
 public:
  explicit Net(const NetParameter& param, Net<Dtype>* memory_share_net = NULL);
  explicit Net(const string& param_file, Net<Dtype>* memory_share_net = NULL);
  virtual ~Net() {}

  // Initialize a network with the network parameter.  If memory_share_net is
  // non-null, any top/bottom blob in this net with an identically-named blob
  // in memory_share_net will share its memory location to save on memory, using
  // memory proportional to max(net_a_blob_size, net_b_blob_size) rather than
  // (net_a_blob_size + net_b_blob_size).
  void Init(const NetParameter& param, Net<Dtype>* memory_share_net = NULL);
  // Run forward with the input blobs already fed separately. You can get the
  // input blobs using input_blobs().
  const vector<Blob<Dtype>*>& ForwardPrefilled(Dtype* loss = NULL);
  // Run forward using a set of bottom blobs, and return the result.
  const vector<Blob<Dtype>*>& Forward(const vector<Blob<Dtype>* > & bottom,
      Dtype* loss = NULL);
  // Run forward using a serialized BlobProtoVector and return the result
  // as a serialized BlobProtoVector
  string Forward(const string& input_blob_protos, Dtype* loss = NULL);

  // The network backward should take no input and output, since it solely
  // computes the gradient w.r.t the parameters, and the data has already
  // been provided during the forward pass.
  void Backward();

  Dtype ForwardBackward(const vector<Blob<Dtype>* > & bottom) {
    Dtype loss;
    Forward(bottom, &loss);
    Backward();
    return loss;
  }

  // Updates the network weights based on the diff values computed.
  void Update();

  // For an already initialized net, ShareTrainedLayersWith() implicitly copies
  // (i.e., using no additional memory) the already trained layers from another
  // Net.
  void ShareTrainedLayersWith(Net* other);
  // For an already initialized net, CopyTrainedLayersFrom() copies the already
  // trained layers from another net parameter instance.
  void CopyTrainedLayersFrom(const NetParameter& param);
  void CopyTrainedLayersFrom(const string trained_filename);
  // Writes the net to a proto.
  void ToProto(NetParameter* param, bool write_diff = false);

  // returns the network name.
  inline const string& name() { return name_; }
  // returns the layer names
  inline const vector<string>& layer_names() { return layer_names_; }
  // returns the blob names
  inline const vector<string>& blob_names() { return blob_names_; }
  // returns the blobs
  inline const vector<shared_ptr<Blob<Dtype> > >& blobs() { return blobs_; }
  // returns the layers
  inline const vector<shared_ptr<Layer<Dtype> > >& layers() { return layers_; }
  // returns the bottom and top vecs for each layer - usually you won't need
  // this unless you do per-layer checks such as gradients.
  inline vector<vector<Blob<Dtype>*> >& bottom_vecs() { return bottom_vecs_; }
  inline vector<vector<Blob<Dtype>*> >& top_vecs() { return top_vecs_; }
  // returns the parameters
  inline vector<shared_ptr<Blob<Dtype> > >& params() { return params_; }
  // returns the parameter learning rate multipliers
  inline vector<float>& params_lr() {return params_lr_; }
  inline vector<float>& params_weight_decay() { return params_weight_decay_; }
  // Input and output blob numbers
  inline int num_inputs() { return net_input_blobs_.size(); }
  inline int num_outputs() { return net_output_blobs_.size(); }
  inline vector<Blob<Dtype>*>& input_blobs() { return net_input_blobs_; }
  inline vector<Blob<Dtype>*>& output_blobs() { return net_output_blobs_; }
  // has_blob and blob_by_name are inspired by
  // https://github.com/kencoken/caffe/commit/f36e71569455c9fbb4bf8a63c2d53224e32a4e7b
  // Access intermediary computation layers, testing with centre image only
  bool has_blob(const string& blob_name);
  const shared_ptr<Blob<Dtype> > blob_by_name(const string& blob_name);
  bool has_layer(const string& layer_name);
  const shared_ptr<Layer<Dtype> > layer_by_name(const string& layer_name);
  const map<string, int>& layer_names_index() { return layer_names_index_; }
  // Access blob metadata, mainly for testing purposes
  const vector<bool>& blob_need_backward() { return blob_need_backward_; }
  const map<string, int>& blob_names_index() { return blob_names_index_; }
  const vector<pair<int, int> >& blob_top_index() { return blob_top_index_; }
  const vector<vector<pair<int, int> > >& blob_bottom_indices() {
    return blob_bottom_indices_;
  }
  const vector<vector<bool> >& bottom_diff_scales() {
    return bottom_diff_scales_;
  }
  const map<string, int>& param_names_index() { return param_names_index_; }

 protected:
  // Helpers for Init.
  // Append a new input or top blob to the net.
  int AppendTop(const NetParameter& param, const int layer_id,
                const int top_id, Net<Dtype>* memory_share_net);
  // Append a new bottom blob to the net.
  int AppendBottom(const NetParameter& param, const int layer_id,
                   const int bottom_id);
  // Decide whether to do forward/backward computation "in-place".
  void SetUpInPlace(const int layer_id);
  // Function to get misc parameters, e.g. the learning rate multiplier and
  // weight decay.
  void GetLearningRateAndWeightDecay();
  // Make a unique internal blob name from a non-unique user blob name.
  void CanonicalBlobName(const size_t max_chars, const char* user_blob_name,
      const char* layer_name, const int top_blob_index, const int num_top,
      char* canonical_blob_name);

  // Individual layers in the net
  vector<shared_ptr<Layer<Dtype> > > layers_;
  vector<string> layer_names_;
  map<string, int> layer_names_index_;
  vector<bool> layer_need_backward_;
  // blobs stores the blobs that store intermediate results between the
  // layers.
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  // blob_names_ lists the canonical blob names, of the form:
  //   <user blob name>__<layer name>_<layer index>_<top blob index>
  vector<string> blob_names_;
  map<string, int> blob_names_index_;
  vector<string> user_blob_names_;
  // blob_name_to_current_index maps user_blob_name -> blob_idx
  // This changes throughout processing of the NetParameter, as a user_blob_name
  // can be duplicated -- used as the name of multiple input/top blobs.  When
  // we see an input/top blob name duplicated, when it is used as a bottom blob,
  // the name is assumed to refer to the LAST input/top blob specified.
  map<string, int> user_blob_name_to_current_index_;
  vector<bool> blob_need_backward_;
  // blob_idx_to_bottom_idx maps blob_idx -> (layer_idx, top_idx)
  vector<pair<int, int> > blob_top_index_;
  // blob_idx_to_bottom_idx maps blob_idx ->
  //     [ (layer_idx_1, bottom_idx_1), (layer_idx_2, bottom_idx_2), ... ]
  vector<vector<pair<int, int> > > blob_bottom_indices_;
  // bottom_vecs stores the vectors containing the input for each layer.
  // They don't actually host the blobs (blobs_ does), so we simply store
  // pointers.
  vector<vector<Blob<Dtype>*> > bottom_vecs_;
  vector<vector<int> > bottom_id_vecs_;
  vector<vector<bool> > bottom_diff_scales_;
  vector<vector<bool> > bottom_need_backward_;
  // top_vecs stores the vectors containing the output for each layer
  vector<vector<Blob<Dtype>*> > top_vecs_;
  vector<vector<int> > top_id_vecs_;
  vector<int> param_owners_;
  vector<pair<int, int> > param_net_indices_;
  map<string, int> param_names_index_;
  // blob indices for the input and the output of the net
  vector<int> net_input_blob_indices_;
  vector<Blob<Dtype>*> net_input_blobs_;
  vector<Blob<Dtype>*> net_output_blobs_;
  string name_;
  // The parameters in the network.
  vector<shared_ptr<Blob<Dtype> > > params_;
  // the learning rate multipliers
  vector<float> params_lr_;
  // the weight decay multipliers
  vector<float> params_weight_decay_;
  // the bytes of memory used by this net
  size_t memory_used_;
  // available_blobs contains the input/top blobs that have not been used as a
  // bottom blob.
  set<int> available_blobs_;
  DISABLE_COPY_AND_ASSIGN(Net);
};


}  // namespace caffe

#endif  // CAFFE_NET_HPP_
