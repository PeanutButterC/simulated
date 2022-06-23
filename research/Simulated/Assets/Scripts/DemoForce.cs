using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class DemoForce : MonoBehaviour
{
    [SerializeField] float force = 500f;

    Rigidbody rigidBody;

    void Awake() {
        rigidBody = GetComponent<Rigidbody>();
    }

    void Start() {
        rigidBody.AddForce(Vector3.left * force);
    }
}
