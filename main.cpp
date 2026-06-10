#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
const int MAX_MODEL_VAR_INPUT = 2;
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>

class matrix {
public:
  int rows, cols;
  std::vector<float> data;
  matrix(int r, int c) : rows(r), cols(c), data(r * c) {};

  float operator()(int i, int j) const { return data[j + i * cols]; };
  float &operator()(int i, int j) { return data[j + i * cols]; };

  void mat_fill(float x);
  void mat_scale(float x);
  void mat_fill_rand();
};

struct model_var;
struct model_context;
struct model_desc;
struct model_program;

bool mat_add(matrix &out, const matrix &a, const matrix &b) {
  if (a.cols != b.cols || a.rows != b.rows || out.cols != a.cols ||
      out.rows != a.rows) {
    return false;
  }
  int size = out.cols * out.rows;

  for (int i = 0; i < size; i++) {
    out.data[i] = a.data[i] + b.data[i];
  }
  return true;
}

bool mat_sub(matrix &out, const matrix &a, const matrix &b) {
  if (a.cols != b.cols || a.rows != b.rows || out.cols != a.cols ||
      out.rows != a.rows) {
    return false;
  }
  int data = out.cols * out.rows;
  for (int i = 0; i < data; i++) {
    out.data[i] = a.data[i] - b.data[i];
  }
  return true;
}

bool mat_mul_nn(matrix &out, const matrix &a, const matrix &b) {
  if (a.cols != b.rows || out.rows != a.rows || out.cols != b.cols) {
    return false;
  }
  for (int i = 0; i < out.rows; i++) {
    for (int k = 0; k < a.cols; k++) {
      for (int j = 0; j < out.cols; j++) {
        out(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return true;
}

bool mat_mul_nt(matrix &out, const matrix &a, const matrix &b) {
  if (a.cols != b.cols || out.rows != a.rows || out.cols != b.rows) {
    return false;
  }
  for (int i = 0; i < out.rows; i++) {
    for (int j = 0; j < out.cols; j++) {
      for (int k = 0; k < a.cols; k++) {
        out(i, j) += a(i, k) * b(j, k);
      }
    }
  }
  return true;
}

bool mat_mul_tn(matrix &out, const matrix &a, const matrix &b) {
  if (a.rows != b.rows || out.rows != a.cols || out.cols != b.cols) {
    return false;
  }
  for (int i = 0; i < out.rows; i++) {
    for (int k = 0; k < a.rows; k++) {
      for (int j = 0; j < out.cols; j++) {
        out(i, j) += a(k, i) * b(k, j);
      }
    }
  }
  return true;
}

bool mat_mul_tt(matrix &out, const matrix &a, const matrix &b) {
  if (a.rows != b.cols || out.rows != a.cols || out.cols != b.rows) {
    return false;
  }
  for (int i = 0; i < out.rows; i++) {
    for (int k = 0; k < a.rows; k++) {
      for (int j = 0; j < out.cols; j++) {
        out(i, j) += a(k, i) * b(j, k);
      }
    }
  }
  return true;
}

bool mat_mul(matrix &out, const matrix &a, const matrix &b, bool zero_out,
             bool transpose_a, bool transpose_b) {
  short transpose = (transpose_a << 1) | transpose_b;
  if (zero_out) {
    out.mat_fill(0.0f);
  }
  switch (transpose) {
  case 0b00:
    return mat_mul_nn(out, a, b);
    break;
  case 0b01:
    return mat_mul_nt(out, a, b);
    break;
  case 0b10:
    return mat_mul_tn(out, a, b);
    break;
  case 0b11:
    return mat_mul_tt(out, a, b);
    break;
  }
  return false;
}

void matrix::mat_fill(float x) {
  int size = this->cols * this->rows;
  for (int i = 0; i < size; i++) {
    this->data[i] = x;
  }
}

void matrix::mat_scale(float x) {
  int size = this->cols * this->rows;
  for (int i = 0; i < size; i++) {
    this->data[i] *= x;
  }
}

float mat_sum(const matrix &mat) {
  int size = mat.cols * mat.rows;
  float sum = 0;
  for (int i = 0; i < size; i++) {
    sum += mat.data[i];
  }

  return sum;
}

bool mat_copy(matrix &out, const matrix &mat) {
  if (mat.cols != out.cols || mat.rows != out.rows) {
    return false;
  }

  int size = out.cols * out.rows;
  for (int i = 0; i < size; i++) {
    out.data[i] = mat.data[i];
  }
  return true;
}

void mat_fill_rand(matrix &out, float bound1, float bound2) {
  int size = out.cols * out.rows;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(bound1, bound2);
  for (int i = 0; i < size; i++) {
    out.data[i] = dist(gen);
  }
}

int mat_arg_max(const matrix &mat) {
  int size = mat.cols * mat.rows;
  float max = -INFINITY;
  int ind = 0;
  for (int i = 0; i < size; i++) {
    if (mat.data[i] > max) {
      ind = i;
      max = mat.data[i];
    }
  }
  return ind;
}

bool mat_relu(matrix &out, const matrix &mat) {
  if (out.cols != mat.cols || out.rows != mat.rows) {
    return false;
  }

  int size = out.cols * out.rows;
  for (int i = 0; i < size; i++) {
    out.data[i] = std::max(0.0f, mat.data[i]);
  }

  return true;
}

bool mat_softmax(matrix &out, const matrix &mat) {
  if (out.rows != mat.rows || out.cols != mat.cols) {
    return false;
  }

  float sum = 0.0f;
  int size = out.cols * out.rows;
  for (int i = 0; i < size; i++) {
    out.data[i] = expf(mat.data[i]);
    sum += out.data[i];
  }

  out.mat_scale(1.0f / sum);
  return true;
}

bool mat_cross_entropy(matrix &out, const matrix &y, const matrix &p) {
  if (y.cols != p.cols || y.rows != p.rows || out.cols != p.cols ||
      out.rows != p.rows) {
    return false;
  }

  int size = out.cols * out.rows;
  for (int i = 0; i < size; i++) {
    out.data[i] = y.data[i] == 0.0f ? 0.0f : -y.data[i] * std::log(p.data[i]);
  }
  return true;
}

enum model_var_flag : uint32_t {
  MV_FLAG_NONE = 0,
  MV_FLAG_REQUIRES_GRAD = 1 << 0,
  MV_FLAG_PARAMETER = 1 << 1,
  MV_FLAG_INPUT = 1 << 2,
  MV_FLAG_OUTPUT = 1 << 3,
  MV_FLAG_DESIRED_OUTPUT = 1 << 4,
  MV_FLAG_COST = 1 << 5,
};

enum class model_var_operation : int {
  Null = 0,
  Relu,
  Softmax,
  Add,
  Sub,
  Matmul,
  Cross_entropy,
};

int get_mv_operatoin_input_number(model_var_operation op) {
  switch (op) {
  case model_var_operation::Null:
    return 0;
    break;
  case model_var_operation::Relu:
    return 1;
    break;
  case model_var_operation::Softmax:
    return 1;
    break;
  case model_var_operation::Add:
    return 2;
    break;
  case model_var_operation::Sub:
    return 2;
    break;
  case model_var_operation::Matmul:
    return 2;
    break;
  case model_var_operation::Cross_entropy:
    return 2;
    break;
  default:
    return 0;
  }
}

std::string get_mv_operatoin_string(model_var_operation op) {
  switch (op) {
  case model_var_operation::Null:
    return "null";
    break;
  case model_var_operation::Relu:
    return "relu";
    break;
  case model_var_operation::Softmax:
    return "softmax";
    break;
  case model_var_operation::Add:
    return "add";
    break;
  case model_var_operation::Sub:
    return "sub";
    break;
  case model_var_operation::Matmul:
    return "matmul";
    break;
  case model_var_operation::Cross_entropy:
    return "cross_entropy";
    break;
  default:
    return "?";
  }
}
struct model_program {
  std::vector<model_var *> vals;
  int num_vals = 0;
};

struct model_context {
  std::vector<std::unique_ptr<model_var>> vals;
  int num_vals = 0;
  model_var *input = nullptr;
  model_var *output = nullptr;
  model_var *desired_output = nullptr;
  model_var *cost = nullptr;
  model_program forward;
  model_program cost_program;
};

struct model_var {
  int index = 0;
  uint flag = 0;
  std::unique_ptr<matrix> val;
  std::unique_ptr<matrix> grad;
  model_var_operation op = model_var_operation::Null;
  model_var *inputs[MAX_MODEL_VAR_INPUT] = {};

  model_var(int rows, int cols, uint flags)
      : val(std::make_unique<matrix>(rows, cols)),
        op(model_var_operation::Null), flag(flags) {
    if (flags & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
      grad = std::make_unique<matrix>(rows, cols);
    }
  }
};

struct model_desc {
  std::unique_ptr<matrix> train_images;
  std::unique_ptr<matrix> test_images;
  std::unique_ptr<matrix> train_lables;
  std::unique_ptr<matrix> test_lables;

  int epochs;
  float learning_rate;
  int batch_size;

  model_desc(matrix *train_images, matrix *test_images, matrix *train_lables,
             matrix *test_lables, int epochs, float learning_rate,
             int batch_size)
      : train_images(train_images), train_lables(train_lables),
        test_images(test_images), test_lables(test_lables), epochs(epochs),
        learning_rate(learning_rate), batch_size(batch_size) {};
};

bool mat_cross_entropy_add_grad(model_var &prev, const model_var &y,
                                const model_var &p) {
  if (y.val->cols != p.val->cols || y.val->rows != p.val->rows) {
    return false;
  }

  int size = p.val->rows * p.val->cols;
  if (y.flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
    for (int i = 0; i < size; i++) {
      y.grad->data[i] += -std::log(p.val->data[i]) * prev.grad->data[i];
    }
  }

  if (p.flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
    for (int i = 0; i < size; i++) {
      p.grad->data[i] += -y.val->data[i] / p.val->data[i] * prev.grad->data[i];
    }
  }
  return true;
}

bool mat_softmax_add_grad(model_var &prev, model_var &z) {
  if (prev.grad->cols != 1 || z.val->cols != 1) {
    return false;
  }

  int size = std::max(z.val->rows, z.val->cols);
  matrix jacobian(size, size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      jacobian(i, j) = (*prev.val)(i, 0) * ((i == j) - (*prev.val)(j, 0));
    }
  }
  mat_mul(*z.grad, jacobian, *prev.grad, 0, 1, 0);
  return true;
}

bool mat_relu_add_grad(model_var &prev, model_var &z) {
  if (z.val->cols != prev.grad->cols || z.val->rows != prev.grad->rows) {
    return false;
  }

  int size = z.val->cols * z.val->rows;

  for (int i = 0; i < size; i++) {
    z.grad->data[i] += z.val->data[i] > 0 ? prev.grad->data[i] : 0.0f;
  }

  return true;
}

model_var *mv_create(model_context &model, int rows, int cols, uint flags) {
  auto var = std::make_unique<model_var>(rows, cols, flags);
  var->index = model.num_vals++;
  model_var *ptr = var.get();
  model.vals.push_back(std::move(var));

  if (flags & model_var_flag::MV_FLAG_INPUT) {
    model.input = ptr;
  };
  if (flags & model_var_flag::MV_FLAG_OUTPUT) {
    model.output = ptr;
  };
  if (flags & model_var_flag::MV_FLAG_DESIRED_OUTPUT) {
    model.desired_output = ptr;
  };
  if (flags & model_var_flag::MV_FLAG_COST) {
    model.cost = ptr;
  };

  return ptr;
}

model_var *unary_operation(model_context &model, model_var &input, uint flags,
                           int rows, int cols, model_var_operation op) {
  if (input.flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
    flags |= model_var_flag::MV_FLAG_REQUIRES_GRAD;
  }

  model_var *out = mv_create(model, rows, cols, flags);
  out->op = op;
  out->inputs[0] = &input;

  return out;
}

model_var *binary_operation(model_context &model, model_var &input1,
                            model_var &input2, uint flags, int rows, int cols,
                            model_var_operation op) {
  if (input1.flag & model_var_flag::MV_FLAG_REQUIRES_GRAD ||
      input2.flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
    flags |= model_var_flag::MV_FLAG_REQUIRES_GRAD;
  }

  model_var *out = mv_create(model, rows, cols, flags);
  out->op = op;
  out->inputs[0] = &input1;
  out->inputs[1] = &input2;

  return out;
}

model_var *mv_relu(model_context &model, model_var &input, uint flags) {
  return unary_operation(model, input, flags, input.val->rows, input.val->cols,
                         model_var_operation::Relu);
}
model_var *mv_softmax(model_context &model, model_var &input, uint flags) {
  return unary_operation(model, input, flags, input.val->rows, input.val->cols,
                         model_var_operation::Softmax);
}
model_var *mv_add(model_context &model, model_var &input1, model_var &input2,
                  uint flags) {
  return binary_operation(model, input1, input2, flags, input1.val->rows,
                          input1.val->cols, model_var_operation::Add);
}
model_var *mv_sub(model_context &model, model_var &input1, model_var &input2,
                  uint flags) {
  return binary_operation(model, input1, input2, flags, input1.val->rows,
                          input1.val->cols, model_var_operation::Sub);
}

model_var *mv_matmul(model_context &model, model_var &input1, model_var &input2,
                     uint flags) {
  return binary_operation(model, input1, input2, flags, input1.val->rows,
                          input2.val->cols, model_var_operation::Matmul);
}
model_var *mv_cross_entropy(model_context &model, model_var &input1,
                            model_var &input2, uint flags) {
  return binary_operation(model, input1, input2, flags, input1.val->rows,
                          input1.val->cols, model_var_operation::Cross_entropy);
}
model_program model_program_create(const model_context &model,
                                   model_var &root) {
  std::vector<model_var *> out;

  std::vector<bool> visited(model.num_vals);

  std::function<void(model_var *)> dfs = [&](model_var *node) {
    if (node == nullptr || node->index >= model.num_vals ||
        visited[node->index]) {
      return;
    }

    visited[node->index] = true;

    for (int i = 0; i < get_mv_operatoin_input_number(node->op); i++) {
      dfs(node->inputs[i]);
    }

    out.push_back(node);
  };
  dfs(&root);
  return model_program{out, (int)out.size()};
}

void model_compile(model_context &model) {
  if (model.output != nullptr) {
    model.forward = model_program_create(model, *model.output);
  }

  if (model.desired_output != nullptr) {
    model.cost_program = model_program_create(model, *model.cost);
  }
}

void create_mnist_model(model_context &model) {
  model_var *input = mv_create(model, 784, 1, MV_FLAG_INPUT);
  model_var *y = mv_create(model, 10, 1, MV_FLAG_DESIRED_OUTPUT);
  model_var *w0 =
      mv_create(model, 16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *w1 =
      mv_create(model, 16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *w2 =
      mv_create(model, 10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  model_var *b0 =
      mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *b1 =
      mv_create(model, 16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
  model_var *b2 =
      mv_create(model, 10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

  float bound0 = sqrtf(6.0f / (784 + 16));
  float bound1 = sqrtf(6.0f / (16 + 16));
  float bound2 = sqrtf(6.0f / (16 + 10));

  mat_fill_rand(*w0->val, -bound0, bound0);
  mat_fill_rand(*w1->val, -bound1, bound1);
  mat_fill_rand(*w2->val, -bound2, bound2);
  b0->val->mat_fill(0.0f);
  b1->val->mat_fill(0.0f);
  b2->val->mat_fill(0.0f);

  model_var *z0_a = mv_matmul(model, *w0, *model.input, 0);
  model_var *z0_b = mv_add(model, *z0_a, *b0, 0);
  model_var *a0 = mv_relu(model, *z0_b, 0);

  model_var *z1_a = mv_matmul(model, *w1, *a0, 0);
  model_var *z1_b = mv_add(model, *z1_a, *b1, 0);
  model_var *r = mv_relu(model, *z1_b, 0);
  model_var *a1 = mv_add(model, *a0, *r, 0);

  model_var *z2_a = mv_matmul(model, *w2, *a1, 0);
  model_var *z2_b = mv_add(model, *z2_a, *b2, 0);

  model_var *p = mv_softmax(model, *z2_b, MV_FLAG_OUTPUT);
  model_var *cost =
      mv_cross_entropy(model, *model.desired_output, *p, MV_FLAG_COST);
}

void model_compute(model_program &program) {
  for (int i = 0; i < program.num_vals; i++) {
    model_var *cur = program.vals[i];
    switch (cur->op) {
    case model_var_operation::Null:;
      break;
    case model_var_operation::Relu:
      mat_relu(*cur->val, *cur->inputs[0]->val);
      break;
    case model_var_operation::Softmax:
      mat_softmax(*cur->val, *cur->inputs[0]->val);
      break;
    case model_var_operation::Cross_entropy:
      mat_cross_entropy(*cur->val, *cur->inputs[0]->val, *cur->inputs[1]->val);
      break;
    case model_var_operation::Add:
      mat_add(*cur->val, *cur->inputs[0]->val, *cur->inputs[1]->val);
      break;
    case model_var_operation::Sub:
      mat_sub(*cur->val, *cur->inputs[0]->val, *cur->inputs[1]->val);
      break;
    case model_var_operation::Matmul:
      mat_mul(*cur->val, *cur->inputs[0]->val, *cur->inputs[1]->val, 1, 0, 0);
      break;
    }
  }
}

void model_compute_grad(model_program &program) {
  for (int i = 0; i < program.num_vals; i++) {
    model_var *cur = program.vals[i];
    if (cur->flag & MV_FLAG_PARAMETER)
      continue;
    if (cur->grad != nullptr) {
      cur->grad->mat_fill(0.0f);
    }
  }
  program.vals[program.num_vals - 1]->grad->mat_fill(1.0f);
  for (int i = program.num_vals - 1; i >= 0; i--) {
    model_var *cur = program.vals[i];
    if (!(cur->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD)) {
      continue;
    }
    model_var *a = cur->inputs[0];
    model_var *b = cur->inputs[1];

    switch (cur->op) {
    case model_var_operation::Null:;
      break;
    case model_var_operation::Relu:
      mat_relu_add_grad(*cur, *a);
      break;
    case model_var_operation::Softmax:
      mat_softmax_add_grad(*cur, *a);
      break;
    case model_var_operation::Cross_entropy:
      mat_cross_entropy_add_grad(*cur, *a, *b);
      break;
    case model_var_operation::Add:
      if (a->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
        mat_add(*a->grad, *a->grad, *cur->grad);
      }
      if (b->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
        mat_add(*b->grad, *b->grad, *cur->grad);
      };
      break;
    case model_var_operation::Sub:
      if (a->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
        mat_sub(*a->grad, *a->grad, *cur->grad);
      }
      if (b->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
        mat_sub(*b->grad, *b->grad, *cur->grad);
      };
      break;
    case model_var_operation::Matmul:
      if (a->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
        bool is_param = a->flag & MV_FLAG_PARAMETER;
        mat_mul(*a->grad, *cur->grad, *b->val, !is_param, 0, 1);
      }
      if (b->flag & model_var_flag::MV_FLAG_REQUIRES_GRAD) {
        bool is_param = b->flag & MV_FLAG_PARAMETER;
        mat_mul(*b->grad, *a->val, *cur->grad, !is_param, 1, 0);
      };
      break;
    }
  }
}

matrix *mat_load(int rows, int cols, std::string name) {
  matrix *out = new matrix(rows, cols);
  int size = rows * cols;
  std::ifstream file(name, std::ios::binary);

  file.seekg(0, std::ios::end);
  int real_size = file.tellg();
  size = std::min(size * (int)sizeof(float), real_size);
  file.seekg(0, std::ios::beg);

  file.read(reinterpret_cast<char *>(out->data.data()), size);

  return out;
}

void model_feedforward(model_context &model) { model_compute(model.forward); }

void model_train(model_context &model, model_desc &desc) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, desc.train_images->rows - 1);
  std::vector<int> training_order;
  for (int i = 0; i < desc.train_images->rows; i++) {
    training_order.push_back(i);
  }

  for (int epoch = 0; epoch < desc.epochs; epoch++) {
    for (int i = 0; i < desc.train_images->rows; i++) {
      int a = dist(gen);
      int b = dist(gen);
      int tmp = training_order[a];
      training_order[a] = training_order[b];
      training_order[b] = tmp;
    }
    int count = 0;
    for (int i = 0; i < training_order.size(); i++) {
      int input_size = desc.train_images->cols;
      for (int j = 0; j < input_size; j++) {
        model.input->val->data[j] = (*desc.train_images)(training_order[i], j);
      }

      model.desired_output->val->mat_fill(0.0f);
      int desired_output = (int)(*desc.train_lables)(training_order[i], 0);
      model.desired_output->val->data[desired_output] = 1.0f;

      model_feedforward(model);

      model_compute_grad(model.cost_program);

      count++;

      if (count % desc.batch_size != 0) {
        continue;
      }
      for (int val = 0; val < model.num_vals; val++) {
        if (!(model.vals[val]->flag & model_var_flag::MV_FLAG_PARAMETER)) {
          continue;
        }

        int size = model.vals[val]->val->cols * model.vals[val]->val->rows;
        for (int element = 0; element < size; element++) {
          model.vals[val]->val->data[element] -=
              desc.learning_rate * model.vals[val]->grad->data[element] /
              desc.batch_size;
        }
        model.vals[val]->grad->mat_fill(0.0f);
      }
      std::cout << "epochs: " << epoch + 1
                << ", batches: " << count / desc.batch_size + 1 << "\r"
                << std::flush;
    }
  }
}

void draw_mnist_digit(std::vector<float> data) {
  for (int y = 0; y < 28; y++) {
    for (int x = 0; x < 28; x++) {
      float num = data[x + y * 28];
      int col = 232 + static_cast<int>(num * 24);
      std::cout << "\x1b[48;5;" << col << "m  ";
    }
    std::cout << "\n";
  }
  std::cout << "\x1b[0m" << std::endl;
}
int main() {
  matrix *train_images = mat_load(60000, 784, "train_images.mat");
  matrix *test_images = mat_load(10000, 784, "test_images.mat");
  matrix *train_lables = mat_load(60000, 1, "train_labels.mat");
  matrix *test_lables = mat_load(10000, 1, "test_labels.mat");

  // draw_mnist_digit(train_images->data);
  // std::cout << (*train_lables)(0, 0) << "\n";
  // for (int i = 0; i < test_lables->rows; i++) {
  //   std::cout << (*test_lables)(i, 0) << "\n";
  // }
  // return 0;

  model_context model;
  create_mnist_model(model);
  model_compile(model);

  model_desc desc(train_images, test_images, train_lables, test_lables, 10,
                  0.01f, 50);
  model_train(model, desc);

  std::cout << "\n";

  int count = 0;
  for (int i = 0; i < test_images->rows; i++) {
    int input_size = desc.test_images->cols;
    for (int j = 0; j < input_size; j++) {
      model.input->val->data[j] = (*desc.test_images)(i, j);
    }

    model.desired_output->val->mat_fill(0.0f);
    int desired_output = (int)(*desc.test_lables)(i, 0);
    model.desired_output->val->data[desired_output] = 1.0f;

    model_feedforward(model);
    // for (int i = 0; i < 10; i++) {
    //   std::cout << (*model.output->val)(i, 0) << " ";
    // }
    // std::cout << "\n";

    int prediction = mat_arg_max(*model.output->val);
    if (prediction == desired_output) {
      count++;
    }

    std::cout << "total: " << count << "/10000, " << "\r" << std::flush;
  }
  std::cout << "\n";
  std::cout << (float)count / test_images->rows * 100 << " %";
  return 0;
};
