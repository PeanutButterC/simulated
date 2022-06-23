using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace BehaviorDesigner.Runtime.Tasks
{
    public class Act : Action
    {
        public enum Action {
            Pick,
            Put,
            Throw,
            Drop,
            Hit,
            Push
        }

        public Action action;

        ActionController actionController;

        public override void OnAwake()
        {
            actionController = GetComponent<ActionController>();            
        }

        public override void OnStart()
        {
            switch(action) {
                case Action.Pick:
                    actionController.Pick();
                    break;
                case Action.Put:
                    actionController.Put();
                    break;
                case Action.Throw:
                    actionController.Throw();
                    break;
                case Action.Drop:
                    actionController.Drop();
                    break;
                case Action.Hit:
                    actionController.Hit();
                    break;
                case Action.Push:
                    actionController.Push();
                    break;
            }
        }

        public override TaskStatus OnUpdate()
        {
            if(actionController.Pending)
                return TaskStatus.Running;
            else
                return TaskStatus.Success;
        }
    }
}
