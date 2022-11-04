
class GoalTracker:
  weights = "" #file name here

  def __init__(self) -> None:
    self.file = self.weights 
    self.model = self.load_model()


  def load_model(self):
    #TODO create model
    return