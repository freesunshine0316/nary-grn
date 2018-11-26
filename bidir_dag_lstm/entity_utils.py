
import tensorflow as tf

def collect_by_indices(memory, indices): # [batch, node_num, dim], [batch, 3, 5]
    batch_size = tf.shape(indices)[0]
    entity_num = tf.shape(indices)[1]
    entity_size = tf.shape(indices)[2]
    idxs = tf.range(0, limit=batch_size) # [batch]
    idxs = tf.reshape(idxs, [-1, 1, 1]) # [batch, 1, 1]
    idxs = tf.tile(idxs, [1, entity_num, entity_size])
    indices = tf.maximum(indices, tf.zeros_like(indices, dtype=tf.int32))
    indices = tf.stack((idxs,indices), axis=3) # [batch,3,5,2]
    return tf.gather_nd(memory, indices)

class Entity:
    # encoder_states [batch, node_num, encoder_dim]
    def __init__(self, encoder_states):
        self.entity_indices = tf.placeholder(tf.int32, [None, None, None],
                name="entity_indices")
        self.entity_indices_mask = tf.placeholder(tf.float32, [None, None, None],
                name="entity_indices_mask")

        batch_size = tf.shape(encoder_states)[0]
        node_num = tf.shape(encoder_states)[1]
        dim = tf.shape(encoder_states)[2]
        entity_num = tf.shape(self.entity_indices)[1]
        entity_size = tf.shape(self.entity_indices)[2]

        # entity_states [batch, 3, 5, dim]
        self.entity_states = collect_by_indices(encoder_states, self.entity_indices)
        # applying mask
        self.entity_states = self.entity_states * tf.expand_dims(self.entity_indices_mask, axis=-1)
        # average within each entity: [batch, 3, encoder_dim]
        self.entity_states = tf.reduce_mean(self.entity_states, axis=2)
