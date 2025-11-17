import tensorflow as tf


class SSDLoss(object):
    def __init__(self, num_classes=2, alpha=1.0, neg_pos_ratio=3.0, background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.n_neg_min = 0
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard
    
    def smooth_L1_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)
    
    def log_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-7)
        softmax_loss = -tf.reduce_sum(y_true * tf.compat.v1.log(y_pred),axis=-1)
        # softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),axis=-1)
        return softmax_loss
    
    def compute_loss(self, y_true, y_pred):
        self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
        self.n_neg_min = tf.constant(self.n_neg_min)
        self.alpha = tf.constant(self.alpha)
        
        batch_size = tf.shape(y_pred)[0]
        n_boxes = tf.shape(y_pred)[1]
        
        # 1. Compute the losses for class and box predictions;
        conf_loss = tf.compat.v1.to_float(self.log_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:]))
        loc_loss = tf.compat.v1.to_float(self.smooth_L1_loss(y_true[:, :, :4], y_pred[:, :, :4]))
        
        # 2: Compute the classification losses for the positive and negative targets.


class MultiboxLoss(object):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0, background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = negatives_for_hard
    
    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
        return tf.reduce_sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = tf.maximum(y_pred, 1e-7)
        softmax_loss = -tf.reduce_sum(y_true * tf.compat.v1.log(y_pred),axis=-1)
        # softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),axis=-1)
        return softmax_loss

    def compute_loss(self, y_true, y_pred):
        num_boxes = tf.compat.v1.to_float(tf.shape(y_true)[1])
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1],
                                       y_pred[:, :, 4:])
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])
        pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -1],
                                     axis=1)
        pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -1],
                                      axis=1)
        num_pos = tf.reduce_sum(y_true[:, :, -1], axis=-1)
        num_neg = tf.minimum(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        pos_num_neg_mask = tf.greater(num_neg, 0)
        has_min = tf.compat.v1.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * self.negatives_for_hard]])
        num_neg_batch = tf.reduce_sum(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.compat.v1.to_int32(num_neg_batch)
        confs_start = 4 + self.background_label_id + 1
        confs_end   = confs_start + self.num_classes - 1
        max_confs = tf.reduce_sum(y_pred[:, :, confs_start:confs_end], axis=2)

        max_confs   = tf.reshape(max_confs * (1 - y_true[:, :, -1]), [-1])
        _, indices  = tf.nn.top_k(max_confs, k=num_neg_batch)

        neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), indices)

        # 进行归一化
        num_pos     = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        total_loss  = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss) + tf.reduce_sum(self.alpha * pos_loc_loss)
        total_loss /= tf.reduce_sum(num_pos)
        return total_loss