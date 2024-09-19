You are a helpful AI coding assistant, who is proficient in python coding and debugging.


We are going to give you a Python program and a mutant diff. We want you to write a test case that detects the mutant. The test case should pass when executed with the correct code, but fail when executed with the mutant.

- Output the test as single Python function called `test__<name>` with no parameters.
- Don't use any testing frameworks.
- Put your code in a markdown block and specify the language.
- Import all necessary files in your test code. You can assume that all python files we give you are in the root directory.
- Use assertions where appropriate.

Example:

```python
from sieve import sieve

def test_sieve()
    # test code here
```

Some mutants may be equivalent. Equivalent mutants don't change the behavior of the code, so they cannot be detected by a test. An example is changing `x=a+b` to `x=b+a`. If you believe a mutant to be equivalent, please claim the mutant as equivalent by writing the `# Equivalent Mutant` headline and giving a short description of why you think the mutant is equivalent.


# Task

```python reverse_linked_list.py
01  """
02  Reverse Linked List
03
04  Reverses a linked list and returns the new head.
05
06  Input:
07      node: The head of a singly-linked list
08
09  Precondition:
10      The input is acyclic
11
12  Side effect:
13      Mutates the list nodes' successor pointers
14
15  Output:
16      The head of the reversed linked list
17  """
18
19  def reverse_linked_list(node):
20      prevnode = None
21      while node:
22          nextnode = node.successor
23          node.successor = prevnode
24          prevnode = node
25          node = nextnode
26      return prevnode
```

node.py:
```python node.py
class Node:
    def __init__(self, value=None, successor=None, successors=[], predecessors=[], incoming_nodes=[], outgoing_nodes=[]):
        self.value = value
        self.successor = successor
        self.successors = successors
        self.predecessors = predecessors
        self.incoming_nodes = incoming_nodes
        self.outgoing_nodes = outgoing_nodes

    def successor(self):
        return self.successor

    def successors(self):
        return self.successors

    def predecessors(self):
        return self.predecessors
```

```diff mutant.diff
diff --git a/reverse_linked_list.py b/reverse_linked_list.py
index 325641c..94b6dd3 100644
--- a/reverse_linked_list.py
+++ b/reverse_linked_list.py
@@ -21,6 +21,5 @@ def reverse_linked_list(node):
     while node:
         nextnode = node.successor
         node.successor = prevnode
-        prevnode = node
         node = nextnode
     return prevnode
```

# Test


Here's the test function that will detect the mutant in the `reverse_linked_list.py` file. The mutant code removes the assignment `prevnode = node`, which will cause the reversed linked list to not form correctly. The test we create will check if the linked list is reversed properly.

```python
from reverse_linked_list import reverse_linked_list
from node import Node

def test_reverse_linked_list():
    # Create a linked list: 1 -> 2 -> 3
    node3 = Node(3)
    node2 = Node(2, node3)
    node1 = Node(1, node2)

    # Reverse the linked list
    new_head = reverse_linked_list(node1)

    # Check the new head
    assert new_head.value == 3
    assert new_head.successor.value == 2
    assert new_head.successor.successor.value == 1
    assert new_head.successor.successor.successor is None

    # Verify the original head (node1) now points to None
    assert node1.successor is None
```

This test creates a simple linked list of three nodes, reverses it, and then checks if the new head and the order of nodes are as expected after the reversal. This test should pass with the original code and fail with the provided mutant, which omits the crucial step of updating `prevnode`.


# Results

The LLM found a test case that detects the mutant.

## Test Case

```python
from reverse_linked_list import reverse_linked_list
from node import Node

def test_reverse_linked_list():
    # Create a linked list: 1 -> 2 -> 3
    node3 = Node(3)
    node2 = Node(2, node3)
    node1 = Node(1, node2)

    # Reverse the linked list
    new_head = reverse_linked_list(node1)

    # Check the new head
    assert new_head.value == 3
    assert new_head.successor.value == 2
    assert new_head.successor.successor.value == 1
    assert new_head.successor.successor.successor is None

    # Verify the original head (node1) now points to None
    assert node1.successor is None
```

## Test on correct code

```

```

## Test on mutant

```
Traceback (most recent call last):
File "test.py", line 22, in <module>
test_reverse_linked_list()
File "test.py", line 14, in test_reverse_linked_list
assert new_head.value == 3
^^^^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'value'
```
The test exited with exit code 1.
