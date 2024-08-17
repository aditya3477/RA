#!/usr/bin/env python3
import rospy
import numpy as np
import math
import matplotlib.pyplot as plt

from geometry_msgs.msg import Twist
from gazebo_msgs.msg import LinkStates
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

# Global Constants
COST_FOR_DIAGONAL_MOVEMENT = math.sqrt(2)
COST_FOR_LINEAR_MOVEMENT = 1

EPSILON = 1
OFFSET = 0.5
PLOT_VISIBILITY = 5
DIRECTIONS = (
  (-1,0),(0,-1),(1,0),(0,1), # => LINEAR MOVEMENT
  (-1,-1),(1,-1),(1,1),(-1,1), # => DIAGONAL MOVEMENT
)
MAP_RESOLUTION = (0.5,0.5)
MAP = [
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  [1,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
  [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
  [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
  [1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
  [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1],
  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

def costFn(fCost:float,hCost:float) -> float:
  return fCost + EPSILON * hCost

def euclidean(point1,point2) -> float:
  # https://www.geeksforgeeks.org/calculate-the-euclidean-distance-using-numpy/
  return np.linalg.norm(point1 - point2)

def diagonalAndLinear(point1, point2) -> float:
  diffInX = abs(point1[0] - point2[0])
  diffInY = abs(point1[1] - point2[1])
  if diffInX < diffInY:
    return \
      COST_FOR_DIAGONAL_MOVEMENT * diffInX + \
      COST_FOR_LINEAR_MOVEMENT * (diffInY - diffInX)
  else:
    return \
      COST_FOR_DIAGONAL_MOVEMENT * diffInY + \
      COST_FOR_LINEAR_MOVEMENT * (diffInX - diffInY)

def extractInfo(botRefFrame,state):
  index = state.name.index(botRefFrame)
  pose = state.pose[index]
  position = [round(pose.position.x), round(pose.position.y)]
  yaw = round(euler_from_quaternion([
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z,
    pose.orientation.w
  ])[2])
  return (position,yaw)

class Node:
  position = []
  gCost : float = 0
  hCost : float = 0
  parent = None

  def __init__(self, position) -> None:
    self.position = np.array(position)
    pass

  @property
  def fCost(self) -> float:
    return costFn(self.gCost,self.hCost)

  def __eq__(self, object: object) -> bool:
    if isinstance(object, Node):
      return np.array_equal(self.position,object.position)
    return False

class AStarAlgorithm:
  def __init__(self, publisherInfo, heuristicFn, map, mapResolution,botRefFrame ) -> None:
    self.heuristicFn = heuristicFn
    self.map = np.array(map)
    self.mapResolution = mapResolution
    self.publisher = rospy.Publisher(
      publisherInfo['topic'],
      publisherInfo['dataType'],
      queue_size=10
    )
    self.CACHED_HUSKY_POSE = False
    self.botRefFrame = botRefFrame
    pass

  def extractInfo(self,state):
    return extractInfo(self.botRefFrame,state)
  
  def waitForInitialState(self,stateInfo):
    self.stateInfo = stateInfo
    initialState = rospy.wait_for_message(stateInfo['topic'],stateInfo['dataType'])
    return self.extractInfo(initialState)
  
  def __doesCoordinateHitBoundary(self,x,y) -> bool:
    mapShape = self.map.shape
    return x < 0 or \
           y < 0 or \
           x >= mapShape[0] or \
           y >= mapShape[1]

  def __hasEncounteredObstacle(self,x,y) -> bool:
    map = self.map
    # As the movement towards top is considered as negative Y in node grid
    return map[int(x)][int(map.shape[1] - 1 - y)] == 1

  def __hasObstacleInLinearDirections(self,i,boolArr) -> bool:
    i = i % 4
    linearCombinations = [[0,1],[1,2],[2,3],[3,0]]
    return np.any(boolArr[x] for x in linearCombinations[i])

  def findPath(self,start,goal):
    map = self.map
    start = self.worldToNode([0,0] or self.initPos)
    goal = self.worldToNode(goal)
    if(map[int(goal[0])][int(goal[1])] == 1):
      rospy.logerr("\nGOAL Position points to an Obstacle. Change the Goal Position")
      return 
    heuristicFn = self.heuristicFn
    openNodes = []
    visitedNodes = []
    startNode = Node(start)
    endNode = Node(goal)
    startNode.gCost = 0
    startNode.hCost = heuristicFn(startNode.position,endNode.position)
    openNodes.append(startNode)

    while len(openNodes) > 0:
      minFCost = np.inf
      minHCost = np.inf
      currNode = None
      for node in openNodes:
        if(node.fCost < minFCost or (node.fCost == minFCost and node.hCost < minHCost)):
          minFCost = node.fCost
          minHCost = node.hCost
          currNode = node
      
      openNodes.remove(currNode)
      visitedNodes.append(currNode)
      if heuristicFn(currNode.position,endNode.position) == 0:
        path = []
        traversingNode = currNode
        while True:
          path.append(traversingNode.position)
          if traversingNode.parent is None:
            print("Total Cost to reach the goal", traversingNode.fCost)
            break
          traversingNode = traversingNode.parent
        path.reverse()
        return np.array(path)
      
      obstaclesInLinearMovement = np.full(4,True)

      for i,(relX,relY) in enumerate(DIRECTIONS):
        nextX = currNode.position[0] + relX
        nextY = currNode.position[1] + relY
        nextNode = Node([nextX,nextY])
        if (self.__doesCoordinateHitBoundary(nextX,nextY)):
          continue
        if (self.__hasEncounteredObstacle(nextX,nextY)):
          if i < 4: # Checks for Linear Movement
            obstaclesInLinearMovement[i] = True
          continue
        if i >= 4 and self.__hasObstacleInLinearDirections(i,obstaclesInLinearMovement):
          continue
        if nextNode in visitedNodes:
          continue
        nextNode.gCost = heuristicFn(startNode.position,nextNode.position)
        nextNode.hCost = heuristicFn(nextNode.position,endNode.position)
        nextNode.parent = currNode

        for node in openNodes:
          if nextNode == node:
            if nextNode.gCost < node.gCost:
              node.gCost = nextNode.gCost
        if nextNode not in openNodes:
          openNodes.append(nextNode)

  def __offsetPointToCenter(self,point):
    return np.array([ x+OFFSET if x < 0 else x-OFFSET for x in point ])

  def worldToNode(self,position):
    map = self.map
    mapResolution = self.mapResolution
    midX = map.shape[0]//2
    midY = map.shape[1]//2
    node = np.zeros(2)
    node[0] = midX + (position[0]/mapResolution[0])
    node[1] = midY + -1 * (position[1]/mapResolution[1])
    return node

  def nodeToWorld(self,indices):
    map = self.map
    mapResolution = self.mapResolution
    i = indices[0]
    j = indices[1]
    midX = map.shape[0]//2
    midY = map.shape[1]//2
    position = np.zeros(2)
    position[0] = (i - midX) * mapResolution[0]
    position[1] = -1 * (j - midY) * mapResolution[1]
    return position

  def plot(self,path):
    map = self.map
    worldPath = []
    for coordinates in path:
      worldPath.append(
        tuple(self.__offsetPointToCenter(self.nodeToWorld(coordinates)))
      )

    start = path[0]
    end = path[-1]
    figure = plt.figure()
    figure.set_figwidth(8)
    figure.set_figheight(8)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.xticks(np.arange(0,42,2),np.arange(-21,21,2))
    plt.yticks(np.arange(0,42,2),np.arange(-21,21,2))
    plt.plot(np.where(map==0)[0],np.where(map==0)[1],marker="+",color="#D3D3D3", linestyle=" ")
    plt.plot(np.where(map==1)[0],np.where(map==1)[1],marker="s",color="#a0a0a0", linestyle=" ")
    plt.plot(path[:,0],(map.shape[1] - 1) - path[:,1],marker='.',color="blue",label="path")
    plt.plot(start[0],(map.shape[1] - 1) - start[1],marker='o',color="red",label="start")
    plt.plot(end[0],(map.shape[1] - 1) - end[1],marker='o',color="green",label="end")
    plt.title("A* path")
    plt.legend()
    plt.show(block=False)
    figure.suptitle(f'This graph will close in {PLOT_VISIBILITY} seconds')
    plt.pause(PLOT_VISIBILITY)
    plt.close('all')

  def __cacheHuskyPose(self,state):
    self.huskyPosition,self.huskyYaw = extractInfo(self.botRefFrame,state)
    self.CACHED_HUSKY_POSE = True

  def __cacheHuskyPoseOdom(self,data):
    position = data.pose.pose.position
    orientation = data.pose.pose.orientation
    euler = euler_from_quaternion([
      orientation.x,
      orientation.y,
      orientation.z,
      orientation.w
    ])[2]
    self.huskyPosition = [
      round(position.x),
      round(position.y)
    ]
    self.huskyYaw = euler
    self.CACHED_HUSKY_POSE = True


  def subscribeToHuskyPose(self,subscriberInfo):
    rospy.Subscriber(
      subscriberInfo["topic"],
      subscriberInfo["dataType"],
      self.__cacheHuskyPoseOdom
    )

  def __calcRelDist(self,nextGoal):
    distance = self.heuristicFn(self.huskyPosition,nextGoal)
    return distance

  def __calcRelYaw(self,nextGoal):
    position = self.huskyPosition
    yaw = self.huskyYaw
    diffInY = position[1] - nextGoal[1]
    diffInX = position[0] - nextGoal[0]
    yawOfNextGoal = math.atan2(diffInY,diffInX)
    diffInYaw = yawOfNextGoal - yaw
    relYInYaw = math.sin(diffInYaw)
    relXInYaw = math.cos(diffInYaw)
    relYaw = math.atan2(relYInYaw,relXInYaw)
    relYaw += math.pi
    if relYaw > math.pi:
        relYaw = abs(math.pi - relYaw)
        relYaw -= math.pi
    return relYaw

  def moveHuskyAlongPath(self,path) -> None:
    THRESHOLD_DEGREE_FOR_YAW = 1
    THRESHOLD_FOR_DISTANCE = 0.3
    worldPath = [
      self.__offsetPointToCenter(self.nodeToWorld(node)) 
      for node in path
    ]
    print("A* Star Path as in World Coordinates:\n", np.array(worldPath))
    rate = rospy.Rate(5)
    i = 0
    yawInRadians = math.radians(THRESHOLD_DEGREE_FOR_YAW)
    lenOfPath = len(path)
    currentGoal = worldPath[i]
    print("Current Goal : ", currentGoal)
    while not rospy.is_shutdown():
      twist = Twist()
      if not self.CACHED_HUSKY_POSE:
          continue
      currentGoal = worldPath[i]
      relDist = self.__calcRelDist(currentGoal)
      relYaw = self.__calcRelYaw(currentGoal)
      if relDist <= THRESHOLD_FOR_DISTANCE:
        if i < lenOfPath:
          i += 1
          print("Current Goal : ", worldPath[i])
        else:
          print("Husky reached the goal")
          return
      elif relDist > THRESHOLD_FOR_DISTANCE:
        # Tweak this to get proper movement
        MAX_ANGULAR_VEL = 0.1
        MAX_LINEAR_VEL = 0.5
        ANGULAR_SCALE_FACTOR = 0.5
        LINEAR_SCALE_FACTOR = 0.1
        angularVel = min(MAX_ANGULAR_VEL,abs(relYaw)) / MAX_ANGULAR_VEL
        linearVel = min(MAX_LINEAR_VEL,relDist) / MAX_LINEAR_VEL
        
        if relYaw > yawInRadians:
            twist.angular.z = ANGULAR_SCALE_FACTOR * angularVel
            # print('Orienting Towards Next Goal :', round(math.degrees(twist.angular.z),2),'degrees')
        elif relYaw < -1 * yawInRadians:
            twist.angular.z = -1 * ANGULAR_SCALE_FACTOR * angularVel
            # print('Orienting Towards Next Goal :' , round(math.degrees(twist.angular.z),2),'degrees')
        else:
          print("Oriented Towards Next Goal")
          twist.linear.x = LINEAR_SCALE_FACTOR * linearVel
          # print('Moving Towards Next Goal',twist.linear.x)

      self.publisher.publish(twist)
      rate.sleep()

if __name__ == "__main__":
  try:
    rospy.init_node('astar',anonymous=True)
    rate = rospy.Rate(1)
    goalX = rospy.get_param('~goalX',7.0)
    goalY = rospy.get_param('~goalY',-2.0)
    heuristicFn = rospy.get_param('~heuristicFn','euclidean')
    dictOfHeuristicFns = {
      'euclidean' : euclidean,
      'diagonalAndLinear' :  diagonalAndLinear
    }
    astarAlgo = AStarAlgorithm(
      publisherInfo = {
        "topic" : "/husky_velocity_controller/cmd_vel",
        "dataType" : Twist
      },
      botRefFrame = 'husky::base_link',
      map = MAP,
      mapResolution=MAP_RESOLUTION,
      heuristicFn = dictOfHeuristicFns[heuristicFn]
    )
    rate.sleep()
    start = astarAlgo.waitForInitialState(
      stateInfo = {
        "topic" : "/gazebo/link_states",
        "dataType" : LinkStates
      }
    )
    path = astarAlgo.findPath(
      start = start,
      goal = [ goalX, goalY ]
    )
    if len(path) > 0:
      print("A* Star Path as in Node Indices:\n",path)
      astarAlgo.plot(path)
      astarAlgo.subscribeToHuskyPose(
        subscriberInfo = {
          "topic" : "/odometry/filtered",
          "dataType" : Odometry
        }
      )
      astarAlgo.moveHuskyAlongPath(path)
    rospy.spin()
  except rospy.ROSException:
    pass
