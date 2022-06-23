using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace BehaviorDesigner.Runtime.Tasks
{
    public class StateConditional : Conditional
    {
        public Target target;
        public Target.State state;

        ActionController actionController;

        public override void OnAwake()
        {
            actionController = GetComponent<ActionController>();
        }

        public override TaskStatus OnUpdate()
        {
            if(actionController.Pending)
                return TaskStatus.Failure;
            if(target.state == state)
                return TaskStatus.Success;
            else
                return TaskStatus.Failure;
        }
    }
}
