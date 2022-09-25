from .masked_model import MaskedModel, New_MaskedModel
import tensorflow as tf

class Masked_Linear(MaskedModel):

    def _forward(self, x):
        return tf.matmul(x, self.W_prime*self.Weight_matrix)

    def _get_mse_loss(self, X, W_prime):
        return tf.square(tf.linalg.norm(X - self._forward(X)))

class New_Masked_Linear(New_MaskedModel):

    def _forward(self, x):
        return tf.matmul(x, self.W_prime*self.Weight_matrix)

    def _get_mse_loss(self, X, W_prime):
        return tf.square(tf.linalg.norm(X - self._forward(X)))