#pragma once

namespace blace {
namespace ml_core {
class ProgressCallback {
public:
  ProgressCallback();

  // virtual void callback() = 0;
  virtual void progress(int curr) = 0;

  virtual void checkAbort() = 0;

  virtual void throwCancel();

  virtual void onFinish();

  virtual ~ProgressCallback();

  virtual bool wantsRegularCall();

  int getCounter();

  void setCounter(int counter);

protected:
  int _counter;
};

class EmptyCallback : public ProgressCallback {
private:
public:
  EmptyCallback();

  /*void callback() override {

          return;
  }*/

  void progress(int curr) override;
  void checkAbort() override;
};
} // namespace ml_core
} // namespace blace