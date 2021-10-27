using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace BehaviorDesigner.Runtime.Tasks
{
    public class GetNavPos : Conditional
    {
        public SharedGameObject navPos;

        public string tag = "NavPos";

        public override TaskStatus OnUpdate()
        {
            GameObject[] candidates = GameObject.FindGameObjectsWithTag(tag);
            navPos.Value = candidates[Random.Range(0, candidates.Length)];
            return TaskStatus.Success;
        }
    }
}
