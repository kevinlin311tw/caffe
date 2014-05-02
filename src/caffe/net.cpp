// Copyright 2014 BVLC and contributors.

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::make_pair;
using std::map;
using std::min;
using std::ostringstream;
using std::pair;
using std::set;

const char* kInputLayerName = "_NET_INPUT";
const int kMaxBlobNameChars = 256;

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param, Net<Dtype>* memory_share_net) {
  Init(param, memory_share_net);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Net<Dtype>* memory_share_net) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  Init(param, memory_share_net);
}

// void Net<Dtype>::Init(const NetParameter& param);
// Set up the layers, blobs, and connections between layers/blobs.
//
// A single "canonical" blob name refers to exactly one (layer_id, top_index)
// and 0 or more (layer_id, bottom_id).
//
// A single "user" blob name (as specified in the proto) may refer to 1 or more
// "canonical" blob names.  The "or more" is supported mainly for backwards
// compatibility -- previous versions of Caffe decided whether to do in-place
// computation based on the user's specification of the same name for the top
// top and bottom blobs of a layer, but in-place computation is now
// automatically inferred (see "In-place computation" below), so all top names
// can be made unique for clarity without any extra memory cost.
//
// If a user specifies a blob name as a top blob only once, its canonical name
// will be set to the user-specified name.  If a blob name is specified as a top
// more than once, it will map to canonical names of the form
//     <user-specified blob name>__<layer name>_<layer index>_<top blob index>
// to guarantee uniqueness.
//
//
///// In-place computation
//
// Sometimes a layer's computation can be performed "in-place" -- that is, with
// memory shared between the input (bottom) and output (top).  There are four
// Layer functions specifying attributes that are used to infer whether in-place
// computation can be done -- ForwardReusesBottomData, BackwardReusesTopDiff,
// BackwardUsesBottomData, & BackwardUsesTopData.  These are detailed in
// include/caffe/layer.hpp.
//
//
///// Sharing layer outputs (tops) and weights
//
// There are two situations when the Backward pass is a bit trickier than usual:
// (1) when a top blob is used as a bottom blob multiple times, and (2) when
// a weight blob is shared by two or more layers.  These cases are handled very
// similarly.  The first time a top blob is used by a layer as a bottom blob,
// its Backward method directly computes its gradient into its bottom blob's
// diff field, computing { bottom->diff := my_gradient }.  In later uses, the
// Backward method accumulates its gradient, computing
// { bottom->diff += my_gradient }.
template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& param, Net<Dtype>* memory_share_net) {
  LOG(INFO) << "Initializing net from parameters: " << std::endl
            << param.DebugString();
  // Basically, build all the layers and set up its connections.
  name_ = param.name();
  user_blob_names_.clear();
  blob_names_.clear();
  blob_top_index_.clear();
  blob_bottom_indices_.clear();
  user_blob_name_to_current_index_.clear();
  blob_need_backward_.clear();
  net_input_blob_indices_.clear();
  net_input_blobs_.clear();
  layers_.clear();
  layer_names_.clear();
  net_output_blobs_.clear();
  blob_names_index_.clear();
  layer_names_index_.clear();
  params_.clear();
  memory_used_ = 0;
  // Set up the input blobs
  CHECK_EQ(param.input_size() * 4, param.input_dim_size())
      << "Incorrect bottom blob dimension specifications.";
  for (int top_id = 0; top_id < param.input_size(); ++top_id) {
    const int layer_id = -1;  // inputs have fake layer ID -1
    AppendTop(param, layer_id, top_id);
  }
  DLOG(INFO) << "Memory required for input: " << memory_used_ * sizeof(Dtype);
  // Set up the input and output for each layer.
  int num_layers = param.layers_size();
  bottom_vecs_.resize(num_layers);
  bottom_id_vecs_.resize(num_layers);
  bottom_diff_scales_.resize(num_layers);
  bottom_need_backward_.resize(num_layers);
  top_vecs_.resize(num_layers);
  top_id_vecs_.resize(num_layers);
  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    bottom_vecs_[layer_id].clear();
    bottom_id_vecs_[layer_id].clear();
    bottom_diff_scales_[layer_id].clear();
    bottom_need_backward_[layer_id].clear();
    top_vecs_[layer_id].clear();
    top_id_vecs_[layer_id].clear();
    bool in_place = false;  // TODO: implement in-place computation.
    const LayerParameter& layer_param = param.layers(layer_id);
    layers_.push_back(shared_ptr<Layer<Dtype> >(GetLayer<Dtype>(layer_param)));
    layer_names_.push_back(layer_param.name());
    if (layer_names_index_.find(layer_param.name()) ==
        layer_names_index_.end()) {
      layer_names_index_[layer_names_[layer_id]] = layer_id;
    } else {
      LOG(FATAL) << "Duplicate layer name " << layer_param.name();
    }
    LOG(INFO) << "Creating Layer " << layer_param.name();
    // Go through the layer's inputs
    bool layer_need_backward = param.force_backward();
    for (int bottom_id = 0; bottom_id < layer_param.bottom_size();
         ++bottom_id) {
      int bottom_blob_id = AppendBottom(param, layer_id, bottom_id);
      layer_need_backward |= blob_need_backward_[bottom_blob_id];
    }
    // Go through the layer's outputs
    for (int top_id = 0; top_id < layer_param.top_size(); ++top_id) {
      AppendTop(param, layer_id, top_id);
    }
    // After this layer is connected, set it up.
    // LOG(INFO) << "Setting up " << layer_names_[layer_id];
    layers_[layer_id]->CheckBlobCounts(bottom_vecs_[layer_id],
                                       top_vecs_[layer_id]);
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], &top_vecs_[layer_id]);
    SetUpInPlace(layer_id);
    for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id) {
      LOG(INFO) << "Top shape: " << top_vecs_[layer_id][top_id]->num() << " "
          << top_vecs_[layer_id][top_id]->channels() << " "
          << top_vecs_[layer_id][top_id]->height() << " "
          << top_vecs_[layer_id][top_id]->width() << " ("
          << top_vecs_[layer_id][top_id]->count() << ")";
      if (!in_place)
        memory_used_ += top_vecs_[layer_id][top_id]->count();
    }
    DLOG(INFO) << "Memory required for data: " << memory_used * sizeof(Dtype);
    int blobs_lr_size = layer_param.blobs_lr_size();
    int num_param_blobs = layers_[layer_id]->blobs().size();
    CHECK(blobs_lr_size == num_param_blobs || blobs_lr_size == 0)
        << "Incorrect blobs_lr size: should be either 0 or the same as "
           "the number of the layer's parameter blobs: " << num_param_blobs;
    // Check if this layer needs backward operation itself
    if (blobs_lr_size) {
      for (int param_id = 0; param_id < blobs_lr_size; ++param_id) {
        const bool param_need_backward = layer_param.blobs_lr(param_id) > 0;
        layer_need_backward |= param_need_backward;
        layers_[layer_id]->set_param_propagate_down(param_id,
                                                    param_need_backward);
      }
    } else if (layers_[layer_id]->blobs().size()) {
      // catch: if a layer param does not specify blobs_lr, we should assume the
      // learning rate to be 1. Thus we will need to perform backward.
      layer_need_backward = true;
    }
    int blob_name_size = layer_param.blob_name_size();
    CHECK(blob_name_size == num_param_blobs || blob_name_size == 0)
        << "Incorrect blob_name size: should be either 0 or the same as "
           "the number of the layer's parameter blobs: " << num_param_blobs;
    const int blob_share_mode_size = layer_param.blob_share_mode_size();
    CHECK(blob_share_mode_size == num_param_blobs || blob_share_mode_size == 0)
        << "Incorrect blob_share_mode size: should be either 0 or the same as "
           "the number of the layer's parameter blobs: " << num_param_blobs;
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      string param_name;
      if (blob_name_size) {
        param_name = layer_param.blob_name(param_id);
      }
      const int net_param_id = params_.size();
      params_.push_back(layers_[layer_id]->blobs()[param_id]);
      param_net_indices_.push_back(make_pair(layer_id, param_id));
      if (!blob_name_size || !param_name.size() || (param_name.size() &&
          param_names_index_.find(param_name) == param_names_index_.end())) {
        // This layer "owns" this parameter blob -- it is either anonymous
        // (i.e., not given a param_name) or explicitly given a name that we
        // haven't already seen.
        param_owners_.push_back(-1);
        if (blob_name_size) {
          param_names_index_[param_name] = net_param_id;
        }
      } else {
        // Named param blob with name we've seen before: share params
        const int owner_net_param_id = param_names_index_[param_name];
        param_owners_.push_back(owner_net_param_id);
        const pair<int, int>& owner_index =
            param_net_indices_[owner_net_param_id];
        const int owner_layer_id = owner_index.first;
        const int owner_param_id = owner_index.second;
        LOG(INFO) << "Sharing parameters '" << param_name << "' owned by "
                  << "layer '" << layer_names_[owner_layer_id] << "', param "
                  << "index " << owner_param_id;
        Blob<Dtype>* this_blob = layers_[layer_id]->blobs()[param_id].get();
        Blob<Dtype>* owner_blob =
            layers_[owner_layer_id]->blobs()[owner_param_id].get();
        if (blob_share_mode_size > param_id &&
            (layer_param.blob_share_mode(param_id) ==
             LayerParameter_DimCheckMode_PERMISSIVE)) {
          // Permissive dimension checking -- only check counts are the same.
          CHECK_EQ(this_blob->count(), owner_blob->count())
              << "Shared parameter blobs must have the same count.";
        } else {
          // Strict dimension checking -- all dims must be the same.
          CHECK_EQ(this_blob->num(), owner_blob->num())
              << "Shared parameter blobs must have the same num.";
          CHECK_EQ(this_blob->channels(), owner_blob->channels())
              << "Shared parameter blobs must have the same channels.";
          CHECK_EQ(this_blob->height(), owner_blob->height())
              << "Shared parameter blobs must have the same height.";
          CHECK_EQ(this_blob->width(), owner_blob->width())
              << "Shared parameter blobs must have the same width.";
        }
        layers_[layer_id]->blobs()[param_id]->ShareData(
            *layers_[owner_layer_id]->blobs()[owner_param_id]);
      }
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(layer_need_backward);
    if (layer_need_backward) {
      LOG(INFO) << layer_names_[layer_id] << " needs backward computation.";
      for (int j = 0; j < top_id_vecs_[layer_id].size(); ++j) {
        blob_need_backward_[top_id_vecs_[layer_id][j]] = true;
      }
    } else {
      LOG(INFO) << layer_names_[layer_id]
                << " does not need backward computation.";
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<int>::iterator it = available_blobs_.begin();
      it != available_blobs_.end(); ++it) {
    LOG(INFO) << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[*it].get());
  }
  for (size_t i = 0; i < blob_names_.size(); ++i) {
    blob_names_index_[blob_names_[i]] = i;
  }
  GetLearningRateAndWeightDecay();
  LOG(INFO) << "Network initialization done.";
  LOG(INFO) << "Memory required for data: " << memory_used * sizeof(Dtype);
}


// Helper for Net::Init: add a new input or top blob to the net.  (Inputs have
// layer_id == -1, tops have layer_id >= 0.)
template <typename Dtype>
int Net<Dtype>::AppendTop(const NetParameter& param, const int layer_id,
                          const int top_id) {
  shared_ptr<Blob<Dtype> > blob_pointer;
  if (layer_id == -1) {
    blob_pointer.reset(new Blob<Dtype>(param.input_dim(top_id * 4),
                                       param.input_dim(top_id * 4 + 1),
                                       param.input_dim(top_id * 4 + 2),
                                       param.input_dim(top_id * 4 + 3)));
  } else {
    blob_pointer.reset(new Blob<Dtype>());
  }
  const int blob_id = blobs_.size();
  blobs_.push_back(blob_pointer);
  string user_blob_name;
  const LayerParameter& layer_param = param.layers(layer_id);
  if (layer_id == -1) {
    user_blob_name = param.input(top_id);
  } else {
    user_blob_name = layer_param.top(top_id);
  }
  user_blob_names_.push_back(user_blob_name);
  string blob_name;
  ostringstream canonical_blob_name_display;
  if (user_blob_name_to_current_index_.find(user_blob_name) ==
      user_blob_name_to_current_index_.end()) {
    // First occurrence of this user_blob_name -- let the canonical name simply
    // be user_blob_name.
    blob_name = user_blob_name;
  } else {
    string layer_name = (layer_id == -1) ?
                        kInputLayerName : layer_param.name();
    char* blob_name_c_str = new char[kMaxBlobNameChars];
    CanonicalBlobName(kMaxBlobNameChars, user_blob_name.c_str(),
        layer_name.c_str(), top_id, layer_param.top_size(), blob_name_c_str);
    blob_name = blob_name_c_str;
    canonical_blob_name_display << " (" << blob_name << ")";
  }
  LOG(INFO) << layer_param.name() << " -> " << user_blob_name
            << canonical_blob_name_display.str();
  blob_names_.push_back(blob_name);
  user_blob_name_to_current_index_[user_blob_name] = blob_id;
  blob_top_index_.push_back(make_pair(layer_id, top_id));
  vector<pair<int, int> > bottom_indices;
  blob_bottom_indices_.push_back(bottom_indices);
  blob_need_backward_.push_back(param.force_backward());
  available_blobs_.insert(blob_id);
  memory_used_ += blob_pointer->count();
  if (layer_id == -1) {
    net_input_blob_indices_.push_back(blob_id);
    net_input_blobs_.push_back(blob_pointer.get());
  } else {
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  return blob_id;
}


// Helper for Net::Init: add a new bottom blob to the net.
template <typename Dtype>
int Net<Dtype>::AppendBottom(const NetParameter& param, const int layer_id,
                             const int bottom_id) {
  LayerParameter layer_param(param.layers(layer_id));
  const string& user_blob_name = layer_param.bottom(bottom_id);
  if (user_blob_name_to_current_index_.find(user_blob_name) ==
      user_blob_name_to_current_index_.end()) {
    LOG(FATAL) << "Unknown blob input " << user_blob_name
               << " to layer " << layer_id;
  }
  const int blob_id = user_blob_name_to_current_index_[user_blob_name];
  const string& blob_name = blob_names_[blob_id];
  ostringstream canonical_blob_name_display;
  if (blob_name != user_blob_name) {
    canonical_blob_name_display << " (" << blob_name << ")";
  }
  LOG(INFO) << layer_param.name() << " <- " << user_blob_name
            << canonical_blob_name_display.str();
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  const int current_num_consumers = blob_bottom_indices_[blob_id].size();
  blob_bottom_indices_[blob_id].push_back(make_pair(layer_id, bottom_id));
  if (current_num_consumers) {
    // This isn't the first consumer of this blob; accumulate its gradients.
    bottom_diff_scales_[layer_id].push_back(1);
  } else {
    // This is the first time this blob has been consumed; zero it out
    // before computing the diff.
    bottom_diff_scales_[layer_id].push_back(0);
  }
  available_blobs_.erase(blob_id);
  bool need_backward = param.force_backward() || blob_need_backward_[blob_id];
  bottom_need_backward_[layer_id].push_back(need_backward);
  return blob_id;
}


template <typename Dtype>
void Net<Dtype>::SetUpInPlace(const int layer_id) {
  const int num_bottom = bottom_vecs_[layer_id].size();
  const int num_top = top_vecs_[layer_id].size();
  for (int top_id = 0; top_id < min(num_bottom, num_top); ++top_id) {
    // Decide if we can do in-place computation for the data.
    const int blob_id = top_id_vecs_[layer_id][top_id];
    bool in_place_data = true;
    const int num_bottom = bottom_vecs_[layer_id].size();
    int bottom_blob_id = -1;
    // Check if I can't overwrite my bottom data because I reuse it in my
    // forward pass or use it in my backward pass.
    if (layers_[layer_id]->ForwardReusesBottomData(top_id) ||
        layers_[layer_id]->BackwardUsesBottomData(top_id)) {
      in_place_data = false;
    } else {
      // Lookup the ID of the blob I might overwrite, then check if I can't
      // overwrite it because its count doesn't match mine.
      bottom_blob_id = bottom_id_vecs_[layer_id][top_id];
      if (blobs_[bottom_blob_id]->count() != blobs_[blob_id]->count()) {
        in_place_data = false;
      } else {
        const pair<int, int>& producer_index = blob_top_index_[bottom_blob_id];
        const Layer<Dtype>& producer_layer = *layers_[producer_index.first];
        const int producer_top_id = producer_index.second;
        if (producer_layer.BackwardUsesTopData(producer_top_id)) {
          in_place_data = false;
        } else {
          // Check if I can't overwrite my bottom data because one of my
          // siblings (i.e., another layer that takes the top blob whose data I
          // might overwrite as a bottom blob) needs to reuse it in its forward
          // or backward pass.
          const vector<pair<int, int> >& sibling_blob_indices =
              blob_bottom_indices_[bottom_blob_id];
          for (int i = 0; i < sibling_blob_indices.size(); ++i) {
            const int sibling_layer_id = sibling_blob_indices[i].first;
            const Layer<Dtype>& sibling_layer = *layers_[sibling_layer_id];
            const int sibling_top_id = sibling_blob_indices[i].second;
            if (sibling_layer.ForwardReusesBottomData(sibling_top_id) ||
                sibling_layer.BackwardUsesBottomData(sibling_top_id)) {
              in_place_data = false;
            }
          }
        }
      }
    }
    if (in_place_data) {
      LOG(INFO) << "Doing in-place forward computation.";
      CHECK_GE(bottom_blob_id, 0);
      blobs_[blob_id]->ShareData(*blobs_[bottom_blob_id]);
    }
    // Decide if we can do in-place computation for the diff.
    bool in_place_diff = true;
    if (top_id >= num_bottom ||
        layers_[layer_id]->BackwardReusesTopDiff(top_id)) {
      in_place_diff = false;
    } else {
      if (bottom_blob_id < 0) {
        bottom_blob_id = bottom_id_vecs_[layer_id][top_id];
      }
      if (blobs_[bottom_blob_id]->count() != blobs_[blob_id]->count()) {
        in_place_diff = false;
      }
    }
    if (in_place_diff) {
      LOG(INFO) << "Doing in-place backward computation.";
      CHECK_GE(bottom_blob_id, 0);
      blobs_[blob_id]->ShareDiff(*blobs_[bottom_blob_id]);
    }
  }
}


template <typename Dtype>
void Net<Dtype>::CanonicalBlobName(const size_t max_chars,
    const char* user_blob_name, const char* layer_name,
    const int top_blob_index, const int num_top, char* canonical_blob_name) {
  size_t num_chars;
  if (num_top > 1) {
    // Layer with multiple top blobs: canonical name is
    //     <user_blob_name>__<layer_name>_<top_index>
    num_chars = snprintf(canonical_blob_name, max_chars, "%s__%s_%d",
                         user_blob_name, layer_name, top_blob_index);
  } else {
    // Layer with just one top blob: canonical name is
    //     <user_blob_name>__<layer_name>
    num_chars = snprintf(canonical_blob_name, max_chars, "%s__%s",
                         user_blob_name, layer_name);
  }
  // Check that the blob name was not truncated.
  CHECK_LE(num_chars, max_chars);
}


template <typename Dtype>
void Net<Dtype>::GetLearningRateAndWeightDecay() {
  LOG(INFO) << "Collecting Learning Rate and Weight Decay.";
  for (int i = 0; i < layers_.size(); ++i) {
    vector<shared_ptr<Blob<Dtype> > >& layer_blobs = layers_[i]->blobs();
    // push the learning rate mutlipliers
    if (layers_[i]->layer_param().blobs_lr_size()) {
      CHECK_EQ(layers_[i]->layer_param().blobs_lr_size(), layer_blobs.size());
      for (int j = 0; j < layer_blobs.size(); ++j) {
        float local_lr = layers_[i]->layer_param().blobs_lr(j);
        CHECK_GE(local_lr, 0.);
        params_lr_.push_back(local_lr);
      }
    } else {
      for (int j = 0; j < layer_blobs.size(); ++j) {
        params_lr_.push_back(1.);
      }
    }
    // push the weight decay multipliers
    if (layers_[i]->layer_param().weight_decay_size()) {
      CHECK_EQ(layers_[i]->layer_param().weight_decay_size(),
          layer_blobs.size());
      for (int j = 0; j < layer_blobs.size(); ++j) {
        float local_decay = layers_[i]->layer_param().weight_decay(j);
        CHECK_GE(local_decay, 0.);
        params_weight_decay_.push_back(local_decay);
      }
    } else {
      for (int j = 0; j < layer_blobs.size(); ++j) {
        params_weight_decay_.push_back(1.);
      }
    }
  }
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::ForwardPrefilled(Dtype* loss) {
  if (loss != NULL) {
    *loss = Dtype(0.);
  }
  for (int i = 0; i < layers_.size(); ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], &top_vecs_[i]);
    if (loss != NULL) {
      *loss += layer_loss;
    }
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom, Dtype* loss) {
  // Copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return ForwardPrefilled(loss);
}

template <typename Dtype>
string Net<Dtype>::Forward(const string& input_blob_protos, Dtype* loss) {
  BlobProtoVector blob_proto_vec;
  if (net_input_blobs_.size()) {
    blob_proto_vec.ParseFromString(input_blob_protos);
    CHECK_EQ(blob_proto_vec.blobs_size(), net_input_blobs_.size())
        << "Incorrect input size.";
    for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
      net_input_blobs_[i]->FromProto(blob_proto_vec.blobs(i));
    }
  }
  ForwardPrefilled(loss);
  blob_proto_vec.Clear();
  for (int i = 0; i < net_output_blobs_.size(); ++i) {
    net_output_blobs_[i]->ToProto(blob_proto_vec.add_blobs());
  }
  string output;
  blob_proto_vec.SerializeToString(&output);
  return output;
}


template <typename Dtype>
void Net<Dtype>::Backward() {
  for (int i = layers_.size() - 1; i >= 0; --i) {
    if (layer_need_backward_[i]) {
      layers_[i]->AccumBackward(top_vecs_[i], bottom_need_backward_[i],
                                bottom_diff_scales_[i], &bottom_vecs_[i]);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ShareTrainedLayersWith(Net* other) {
  int num_source_layers = other->layers().size();
  for (int i = 0; i < num_source_layers; ++i) {
    Layer<Dtype>* source_layer = other->layers()[i].get();
    const string& source_layer_name = other->layer_names()[i];
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      DLOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer->blobs().size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      Blob<Dtype>* source_blob = source_layer->blobs()[j].get();
      CHECK_EQ(target_blobs[j]->num(), source_blob->num());
      CHECK_EQ(target_blobs[j]->channels(), source_blob->channels());
      CHECK_EQ(target_blobs[j]->height(), source_blob->height());
      CHECK_EQ(target_blobs[j]->width(), source_blob->width());
      target_blobs[j]->ShareData(*source_blob);
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layers_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layers(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      DLOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      CHECK_EQ(target_blobs[j]->num(), source_layer.blobs(j).num());
      CHECK_EQ(target_blobs[j]->channels(), source_layer.blobs(j).channels());
      CHECK_EQ(target_blobs[j]->height(), source_layer.blobs(j).height());
      CHECK_EQ(target_blobs[j]->width(), source_layer.blobs(j).width());
      target_blobs[j]->FromProto(source_layer.blobs(j));
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_input(blob_names_[net_input_blob_indices_[i]]);
  }
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layers();
    for (int j = 0; j < bottom_id_vecs_[i].size(); ++j) {
      layer_param->add_bottom(blob_names_[bottom_id_vecs_[i][j]]);
    }
    for (int j = 0; j < top_id_vecs_[i].size(); ++j) {
      layer_param->add_top(blob_names_[top_id_vecs_[i][j]]);
    }
    layers_[i]->ToProto(layer_param, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::Update() {
  // First, accumulate the diffs of any shared parameters into their owner's
  // diff.
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] >= 0) {
      const int count = params_[i]->count();
      const Dtype* this_diff;
      Dtype* owner_diff;
      switch (Caffe::mode()) {
      case Caffe::CPU:
        this_diff = params_[i]->cpu_diff();
        owner_diff = params_[param_owners_[i]]->mutable_cpu_diff();
        caffe_add(count, this_diff, owner_diff, owner_diff);
        break;
      case Caffe::GPU:
        this_diff = params_[i]->gpu_diff();
        owner_diff = params_[param_owners_[i]]->mutable_gpu_diff();
        caffe_gpu_axpy(count, Dtype(1), this_diff, owner_diff);
        break;
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
    }
  }
  // Now, update the owned parameters.
  for (int i = 0; i < params_.size(); ++i) {
    if (param_owners_[i] < 0) {
      params_[i]->Update();
    }
  }
}

template <typename Dtype>
bool Net<Dtype>::has_blob(const string& blob_name) {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Blob<Dtype> > Net<Dtype>::blob_by_name(
    const string& blob_name) {
  shared_ptr<Blob<Dtype> > blob_ptr;
  if (has_blob(blob_name)) {
    blob_ptr = blobs_[blob_names_index_[blob_name]];
  } else {
    blob_ptr.reset((Blob<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown blob name " << blob_name;
  }
  return blob_ptr;
}

template <typename Dtype>
bool Net<Dtype>::has_layer(const string& layer_name) {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

template <typename Dtype>
const shared_ptr<Layer<Dtype> > Net<Dtype>::layer_by_name(
    const string& layer_name) {
  shared_ptr<Layer<Dtype> > layer_ptr;
  if (has_layer(layer_name)) {
    layer_ptr = layers_[layer_names_index_[layer_name]];
  } else {
    layer_ptr.reset((Layer<Dtype>*)(NULL));
    LOG(WARNING) << "Unknown layer name " << layer_name;
  }
  return layer_ptr;
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
