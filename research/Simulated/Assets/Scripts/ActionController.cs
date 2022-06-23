using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using BehaviorDesigner.Runtime;

[RequireComponent(typeof(BehaviorTree))]
public class ActionController : MonoBehaviour
{    
    [SerializeField] Transform hand;
    [SerializeField] Transform counter;
    [SerializeField] Target target;
    [SerializeField] PhysicMaterial physicMaterial;

    BehaviorTree behavior;
    public BehaviorTree Behavior {
        get {
            if(behavior == null)
                behavior = GetComponent<BehaviorTree>();
            return behavior;
        }
    }    

    ParameterData data;
    Rigidbody handRigidBody;
    State state;
    Vector3 handStartPos;
    Vector3 handThrowPos;
    Vector3 targetCounterPos;
    Vector3 prepareHitPos;
    Vector3 hitForce;
    Vector3 preparePushPos;
    Vector3 targetPushPos;
    bool pushing;

    public bool Pending => state != State.Idle || target.Moving;

    void Awake() {
        handRigidBody = hand.GetComponent<Rigidbody>();
        handStartPos = hand.localPosition;
    }

    public void Initialize(ParameterData data) {
        this.data = data;
        GameObject[] candidates = GameObject.FindGameObjectsWithTag("WanderPos");
        transform.position = candidates[Random.Range(0, candidates.Length)].transform.position;
        transform.rotation = Quaternion.Euler(0, Random.Range(0, 360f), 0);
        hand.localPosition = handStartPos;
        data.Initialize();
        target.Initialize(data.activeMesh, data.mass, data.drag, data.angularDrag);
        physicMaterial.staticFriction = data.staticFriction;
        physicMaterial.dynamicFriction = data.dynamicFriction;
        physicMaterial.bounciness = data.bounciness;
        Behavior.enabled = true;
    }

    void Update() {
        switch(state) {
            case State.Reach:
                hand.position = Vector3.MoveTowards(hand.position, target.transform.position, Time.deltaTime * data.reachSpeed);
                break;
            case State.Pick:
                hand.localPosition = Vector3.MoveTowards(hand.localPosition, handStartPos, Time.deltaTime * data.pickSpeed);
                if(Vector3.Distance(hand.localPosition, handStartPos) < .01f) {
                    state = State.Idle;
                }
                break;
            case State.Put:
                hand.position = Vector3.MoveTowards(hand.position, targetCounterPos, Time.deltaTime * data.putSpeed);
                break;
            case State.Retract:
                hand.localPosition = Vector3.MoveTowards(hand.localPosition, handStartPos, Time.deltaTime * data.pickSpeed);
                if(Vector3.Distance(hand.localPosition, handStartPos) < .01f) {
                    state = State.Idle;
                }
                break;
            case State.Throw:
                hand.localPosition = Vector3.MoveTowards(hand.localPosition, handThrowPos, Time.deltaTime * data.throwSpeed);
                if(Vector3.Distance(hand.localPosition, handThrowPos) < .01f) {
                    state = State.Retract;
                    ReleaseTarget();
                    Vector3 force = (handThrowPos - handStartPos).normalized * data.throwSpeed * data.throwForce;
                    target.Rigidbody.AddForce(force);
                }
                break;
            case State.PrepareHit:
                hand.position = Vector3.MoveTowards(hand.position, prepareHitPos, Time.deltaTime * data.reachSpeed);
                if(Vector3.Distance(hand.position, prepareHitPos) < .01f) {
                    state = State.Hit;
                    target.OnHandEnter += OnHitHandEnter;
                }
                break;
            case State.Hit:
                hand.position = Vector3.MoveTowards(hand.position, target.transform.position, Time.deltaTime * data.hitSpeed);
                break;
            case State.PreparePush:
                hand.position = Vector3.MoveTowards(hand.position, preparePushPos, Time.deltaTime * data.reachSpeed);
                if(Vector3.Distance(hand.position, preparePushPos) < .01f) {
                    state = State.Push;
                    target.OnHandEnter += OnPushHandEnter;
                }
                break;
            case State.Push:
                hand.position = Vector3.MoveTowards(hand.position, targetPushPos, Time.deltaTime * data.pushSpeed);
                if(pushing) {
                    handRigidBody.velocity = target.Rigidbody.velocity;
                }
                if(Vector3.Distance(hand.position, targetPushPos) < .01f) {
                    state = State.Retract;
                    pushing = false;
                }
                break;
        }
    }

    public Frame WriteFrame(Frame frame) {
        frame.target_posX = target.transform.position.x;
        frame.target_posY = target.transform.position.y;
        frame.target_posZ = target.transform.position.z;
        frame.target_rotX = target.transform.rotation.x;
        frame.target_rotY = target.transform.rotation.y;
        frame.target_rotZ = target.transform.rotation.z;
        frame.target_rotW = target.transform.rotation.w;
        frame.hand_posX = hand.transform.position.x;
        frame.hand_posY = hand.transform.position.y;
        frame.hand_posZ = hand.transform.position.z;
        frame.hand_state = (int)state;
        frame.target_state = (int)target.state;
        return frame;
    }

    public void Pick() {
        state = State.Reach;
        target.OnHandEnter += OnPickHandEnter;
    }

    public void Put() {
        Vector3 dir = counter.position - hand.position;
        dir = new Vector3(dir.x, 0, dir.z).normalized;
        Vector3 tangent = Vector3.Cross(dir, Vector3.up);
        Vector3 origin = hand.position + dir * Random.Range(1f, 1.7f) + tangent * Random.Range(-.3f, .3f) + Vector3.up;
        if(Physics.Raycast(origin, Vector3.down, out RaycastHit hit, float.PositiveInfinity, LayerMask.GetMask("Counter"))) {
            targetCounterPos = hit.point;
            target.OnCounterEnter += OnPutCounterEnter;
            state = State.Put;
        } else {
            Debug.LogError("Failed to find counter put position");
        }
    }

    public void Throw() {
        state = State.Throw;
        handThrowPos = handStartPos + new Vector3(Random.Range(-1f, 1f), Random.Range(-.5f, 1.5f), Random.Range(0f, 1f)).normalized * .5f;
    }

    public void Drop() {
        state = State.Retract;
        ReleaseTarget();
        target.Rigidbody.AddForce(Vector3.down * .01f);
    }

    public void Hit() {
        state = State.PrepareHit;
        Vector3 dir = target.transform.position - hand.position;
        dir = new Vector3(dir.x, 0, dir.z).normalized;
        Vector3 tangent = Vector3.Cross(dir, Vector3.up);
        int sign = Random.Range(0, 2) * 2 - 1;
        prepareHitPos = target.transform.position + tangent * sign * .5f;
        hitForce = ((tangent * sign) + Vector3.up * Random.Range(-1f, 1f)).normalized * -data.hitForce;
    }

    public void Push() {
        state = State.PreparePush;
        Vector3 dir = target.transform.position - hand.position;
        dir = new Vector3(dir.x, 0, dir.z).normalized;
        Vector3 tangent = Vector3.Cross(dir, Vector3.up);
        int sign = Random.Range(0, 2) * 2 - 1;
        Vector3 pushDir = (dir * 2 + tangent * sign).normalized;
        preparePushPos = target.transform.position - pushDir * .5f;
        targetPushPos = target.transform.position + pushDir * .5f;
    }

    public void OnPickHandEnter() {
        target.OnHandEnter -= OnPickHandEnter;
        state = State.Pick;
        HoldTarget();
    }

    public void OnHitHandEnter() {
        target.OnHandEnter -= OnHitHandEnter;
        state = State.Retract;
        target.Rigidbody.AddForce(hitForce);
    }

    public void OnPushHandEnter() {
        target.OnHandEnter -= OnPushHandEnter;
        pushing = true;
    }

    public void OnPutCounterEnter() {
        target.OnCounterEnter -= OnPutCounterEnter;
        state = State.Retract;
        ReleaseTarget();
    }

    void HoldTarget() {
        target.transform.SetParent(hand.transform);
        target.Rigidbody.isKinematic = true;
    }

    void ReleaseTarget() {
        target.transform.SetParent(null);
        target.Rigidbody.isKinematic = false;
    }

    public enum State {
        Idle,
        Reach,
        Pick,
        Put,
        Retract,
        Throw,
        PrepareHit,
        Hit,
        PreparePush,
        Push
    }
}
