#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
const int MAX_MODEL_VAR_INPUT = 2;
#include <cstdint>
#include <functional>
#include <memory>
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

// void matrix::mat_fill_rand();
int mat_arg_max(const matrix &mat) {
  int size = mat.cols * mat.rows;
  float max = -INFINITY;
  int ind = 0;
  for (int i = 0; i < size; i++) {
    if (mat.data[i] > max) {
      ind = i;
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

  out.mat_scale(1.0 / sum);
  return true;
}

bool mat_cross_entropy(matrix &out, const matrix &y, const matrix &p) {
  if (y.cols != p.cols || y.rows != p.rows || out.cols != p.cols ||
      out.rows != p.rows) {
    return false;
  }

  int size = out.cols * out.rows;
  for (int i = 0; i < size; i++) {
    out.data[i] = -y.data[i] * logf(p.data[i]);
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
};

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

void compute_model(model_program &program) {
  for (int i = 0; i < program.num_vals; i++) {
    model_var *cur = program.vals[i];
    switch (cur->op) {
    case model_var_operation::Null:;
      break;
    case model_var_operation::Relu:
      mat_relu(program.vals[i]->val, program.vals[i]->inputs[0]);
    }
  }
}

int main() {
  std::vector<float> v{
      1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 10.0, 11.0, 4.0,
  };
  std::vector<float> w{
      2.0, 9.0, 3.0, 4.0, 5.0, 1.0, 10.0, 11.0, 4.0,
  };
  matrix m{3, 3};
  m.data = v;
  matrix n{3, 3};
  n.data = w;
  matrix out{3, 3};
  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 2; b++) {
      mat_mul(out, m, n, 1, a, b);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          std::cout << out.data[j + i * out.cols] << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
  }

  model_context model;
  create_mnist_model(model);
  model_compile(model);
  for (int i = 0; i < model.forward.num_vals; i++) {
    std::cout << get_mv_operatoin_string(model.forward.vals[i]->op) << " ";
  }
  std::cout << "\n";
  for (int i = 0; i < model.cost_program.num_vals; i++) {
    std::cout << get_mv_operatoin_string(model.cost_program.vals[i]->op) << " ";
  }

  return 0;
};
