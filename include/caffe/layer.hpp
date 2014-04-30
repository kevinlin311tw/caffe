// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

using std::vector;

namespace caffe {

template <typename Dtype>
class Layer {
 public:
  // You should not implement your own constructor. Any set up code should go
  // to SetUp(), where the dimensions of the bottom blobs are provided to the
  // layer.
  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {
      // The only thing we do is to copy blobs if there are any.
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());
          blobs_[i]->FromProto(layer_param_.blobs(i));
        }
      }
    }
  virtual ~Layer() {}
  // SetUp: your function should implement this.
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;

  // Forward and backward wrappers. You should implement the cpu and
  // gpu specific implementations instead, and should not change these
  // functions.
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  // if propagate_down[i], Backward( . , . , . ) computes
  //     bottom[i]->diff := dE/d{bottom[i]}
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom);

  // if propagate_down[i], Backward( . , . , . , . ) computes
  //     bottom[i]->diff := dE/d{bottom[i]} + accum_down[i] * bottom[i]->diff
  // (Equivalent to the previous method if (!accum_down[i]) for all i.)
  inline void AccumBackward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<bool>& accum_down,
      vector<Blob<Dtype>*>* bottom);

  // Returns the vector of blobs.
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  // Returns the layer parameter
  const LayerParameter& layer_param() { return layer_param_; }
  // Writes the layer parameter to a protocol buffer
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  // Returns the layer type name.
  virtual inline const char* LayerType() const { return ""; }

  // These methods can be overwritten to declare that this layer type expects
  // a certain number of blobs as input and output.
  //
  // ExactNum{Bottom,Top}Blobs return a non-negative number to require an exact
  // number of bottom/top blobs; the Min/Max versions return a non-negative
  // number to require a minimum and/or maximum number of blobs.
  // If Exact is specified, neither Min nor Max should be specified, and vice
  // versa.  These methods may not rely on SetUp having been called.
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return -1; }
  virtual inline int MaxBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return -1; }
  virtual inline int MaxTopBlobs() const { return -1; }

  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << LayerType() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << LayerType() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << LayerType() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << LayerType() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << LayerType() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << LayerType() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
  }

  // ForwardReusesBottomData(i) should return true if and only if this layer's
  // Forward method requires bottom[i] and top[i] to point to different memory
  // locations, e.g., in a pooling layer when the bottom is reused to compute
  // multiple top indices in a neighborhood.
  //
  // BackwardUses{Bottom,Top}Data(i) should return true if and only if this
  // layer's Backward method uses the data field (i.e., cpu_data() or
  // gpu_data()) of the ith {bottom,top} blob.
  //
  // It is always safe to skip overriding these methods (i.e., always return
  // true); however, returning false when applicable may allow Caffe to perform
  // this layer or an above layer's computation "in-place" -- using the same
  // blob for bottom & top -- thus saving memory.
  //
  // In any given layer, we will do Forward in-place computation at index i --
  // overwriting bottom blob i's data with the output top blob i's data, where
  // the source of bottom blob i is top blob j in layer "producer_layer":
  // if all of these conditions hold:
  //     ! ForwardReusesBottomData(i)
  //     ! BackwardUsesBottomData(i)
  //     ! producer_layer->BackwardUsesTopData(j) // where producer_layer
  //         // is a pointer to the layer producing the source for my input
  //         // bottom blob i as its top blob j
  //     ! producer_layer->force_no_overwrite_top_data(i)
  // Furthermore, we will do Backward in-place computation at index i --
  // overwriting top blob i's diff with botttom blob i's diff -- if all of these
  // conditions hold:
  //     ! BackwardReusesTopDiff(i)
  //     for each consumer_layer taking my top blob i as their bottom blob j:
  //       ! consumer_layer-> force_no_overwrite_bottom_diff(j)
  virtual inline bool ForwardReusesBottomData(int bottom_index) const {
    return true;
  }
  virtual inline bool BackwardReusesTopDiff(int top_index) const {
    return true;
  }
  virtual inline bool BackwardUsesBottomData(int bottom_index) const {
    return true;
  }
  virtual inline bool BackwardUsesTopData(int top_index) const {
    return true;
  }

  // ElementwiseOnlyComputation should return true if and only if all of this
  // layer's top and bottom blobs have the same count, and for any feature index
  // i, top blob feature i depends only on feature i in the bottom blob(s), as
  // in neuron layers.  This function is used by the GradientChecker -- if
  // overridden to true, we can shortcut most of the computation (and check
  // that the layer indeed performs element-wise computation).
  virtual inline bool ElementwiseOnlyComputation() const { return false; }

  inline void set_param_propagate_down(int index, bool value) {
    if (param_propagate_down_.size() <= index) {
      param_propagate_down_.resize(index + 1);
    }
    param_propagate_down_[index] = value;
  }

 protected:
  // The protobuf that stores the layer parameters
  LayerParameter layer_param_;
  // The vector that stores the parameters as a set of blobs.
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<bool> param_propagate_down_;
  // The vector that stores the parameters to accumulate diffs, for layers that
  // only implement the regular Backward methods.  When possible to do more
  // efficiently than the default Backward (which allocates an extra Blob
  // for the diff, runs the normal Backward method on this extra blob, and then
  // adds the extra blob to the original bottom blob), layers should also
  // implement their own Backward methods, in which case this vector will
  // be empty.
  vector<shared_ptr<Blob<Dtype> > > accum_bottom_blobs_;

  // Forward functions: compute the layer output
  // (and loss layers return the loss; other layers return the dummy value 0.)
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) = 0;
  // If no gpu code is provided, we will simply use cpu code.
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  // Backward functions: compute the gradients for any parameters and
  // bottom blobs for which propagate_down is true.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) = 0;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  // AccumBackward functions: compute the gradients for any parameters and
  // bottom blobs for which propagate_down is true, adding to the current
  // diff if accum_down is true (zeroing out otherwise, as in Backward).
  // These generalize the Backward functions and a default implementation is
  // is provided for layers that implement only the Backward functions.
  virtual void AccumBackward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<bool>& accum_down,
      vector<Blob<Dtype>*>* bottom);
  virtual void AccumBackward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<bool>& accum_down,
      vector<Blob<Dtype>*>* bottom);

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return Forward_cpu(bottom, top);
  case Caffe::GPU:
    return Forward_gpu(bottom, top);
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return Dtype(0);
  }
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
inline void Layer<Dtype>::AccumBackward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<bool>& accum_down,
    vector<Blob<Dtype>*>* bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    AccumBackward_cpu(top, propagate_down, accum_down, bottom);
    break;
  case Caffe::GPU:
    AccumBackward_gpu(top, propagate_down, accum_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

// The layer factory function
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param);

template <typename Dtype>
void Layer<Dtype>::AccumBackward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<bool>& accum_down,
    vector<Blob<Dtype>*>* bottom) {
  const int num_bottom = bottom->size();
  vector<Blob<Dtype>*> orig_bottom(num_bottom);
  for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
    if (accum_down[bottom_id] && propagate_down[bottom_id]) {
      if (accum_bottom_blobs_.size() <= bottom_id) {
        accum_bottom_blobs_.resize(bottom_id + 1);
      }
      if (!accum_bottom_blobs_[bottom_id]) {
        accum_bottom_blobs_[bottom_id].reset(new Blob<Dtype>());
        accum_bottom_blobs_[bottom_id]->ReshapeLike(*(*bottom)[bottom_id]);
      }
      accum_bottom_blobs_[bottom_id]->CopyFrom(*(*bottom)[bottom_id]);
      orig_bottom[bottom_id] = (*bottom)[bottom_id];
      (*bottom)[bottom_id] = accum_bottom_blobs_[bottom_id].get();
    }
  }
  Backward_cpu(top, propagate_down, bottom);
  for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
    if (accum_down[bottom_id] && propagate_down[bottom_id]) {
      (*bottom)[bottom_id] = orig_bottom[bottom_id];
      const int count = (*bottom)[bottom_id]->count();
      const Dtype* accum_diff = accum_bottom_blobs_[bottom_id]->cpu_diff();
      Dtype* diff = (*bottom)[bottom_id]->mutable_cpu_diff();
      caffe_add(count, accum_diff, diff, diff);
    }
  }
}

template <typename Dtype>
void Layer<Dtype>::AccumBackward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<bool>& accum_down,
    vector<Blob<Dtype>*>* bottom) {
  const int num_bottom = bottom->size();
  vector<Blob<Dtype>*> orig_bottom(num_bottom);
  for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
    if (accum_down[bottom_id]) {
      if (accum_bottom_blobs_.size() <= bottom_id) {
        accum_bottom_blobs_.resize(bottom_id + 1);
      }
      if (!accum_bottom_blobs_[bottom_id]) {
        accum_bottom_blobs_[bottom_id].reset(new Blob<Dtype>());
        accum_bottom_blobs_[bottom_id]->ReshapeLike(*(*bottom)[bottom_id]);
      }
      accum_bottom_blobs_[bottom_id]->CopyFrom(*(*bottom)[bottom_id]);
      orig_bottom[bottom_id] = (*bottom)[bottom_id];
      (*bottom)[bottom_id] = accum_bottom_blobs_[bottom_id].get();
    }
  }
  Backward_gpu(top, propagate_down, bottom);
  for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
    if (accum_down[bottom_id]) {
      (*bottom)[bottom_id] = orig_bottom[bottom_id];
      const int count = (*bottom)[bottom_id]->count();
      const Dtype* accum_diff =
          accum_bottom_blobs_[bottom_id]->gpu_diff();
      Dtype* diff = (*bottom)[bottom_id]->mutable_gpu_diff();
      caffe_gpu_axpy(count, Dtype(1), accum_diff, diff);
    }
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
