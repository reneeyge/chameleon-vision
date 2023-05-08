using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class EyeControlAgent : Agent
{
	#region Parameters    
	[Header("Eye control parameters")]
	[Tooltip("Target for the agent to look for with both eyes (cameras).")]
	public GameObject target;

	[Tooltip("Camera used for rendering the left side view.")]
	public Camera leftEyeCamera;

	[Tooltip("Camera used for rendering the right side view.")]
	public Camera rightEyeCamera;

	[Tooltip("Plane used to limit left eye camera rotation.")]
	public GameObject leftEyeRestrictionPlane;

	[Tooltip("Plane used to limit right eye camera rotation.")]
	public GameObject rightEyeRestrictionPlane;

	[Tooltip("Controls the speed at wich the agent is able to rotate each eye, in degrees per second.")]
	[Range(10.0f, 180.0f)]
	public float eyeRotationSpeed;

    [Tooltip("Controls the amount of time the target agent showld keep eyesight on the target before goal.")]
    [Range(0.0f, 10.0f)]
    public float eyeOnTargetTime;

    [Tooltip("Wether the agent should use the camera rotation positions as inputs.")]
	public bool useVectorObservations;

	[Header("On inference")]
	[Tooltip("Time between desicions on inference mode.")]
	[Range(0.05f, 0.5f)]
	public float timeBetweenDecisionsAtInference;
	#endregion

	#region Member Parameters
	/// <summary>
	/// Time since last desicion was taken in inference mode.
	/// </summary>
	float m_TimeSinceDecision;

	/// <summary>
	/// The amount of time the 
	/// </summary>
	float m_TimeOnTarget;

    /// <summary>
    /// Wheter the left camera is fully focused on the target or not.
    /// </summary>
    bool m_LeftEyeWasOnTarget;

    /// <summary>
    /// Wheter the right camera is fully focused on the target or not.
    /// </summary>
    bool m_RightEyeWasOnTarget;

    /// <summary>
    /// Academy training paramters.
    /// </summary>
    EnvironmentParameters m_ResetParameters;

	/// <summary>
	/// The target's mesh renderer.
	/// </summary>
	MeshRenderer m_TargetMeshRenderer;

    /// <summary>
    /// Define the index related to each posible discrete action rotation direction.
    /// </summary>
    enum CameraRotationActions : int
    {
        Negative = 0,
        Nothing = 1,
		Positive = 2,
    }
    #endregion

    #region Agent Overrides
    /// <summary>
    /// Initialice agent parameters and set enviroment.
    /// </summary>
    public override void Initialize()
	{
		// Get enviroment parameters from the academy parameters.
		m_ResetParameters = Academy.Instance.EnvironmentParameters;

		// Get target's mesh renderer.
		m_TargetMeshRenderer = target.GetComponent<MeshRenderer>();

        // Set the previous state for if the agent has either eye on the target to false.
        m_LeftEyeWasOnTarget = false;
        m_RightEyeWasOnTarget = false;

        // Clear the eye sight timer.
        m_TimeOnTarget = 0.0f;

        // Set the agent.
        SetAgent();

		// Set the target.
		SetTarget();		
	}

	/// <summary>
	/// Add vector observations if the useVectorObservations flag is set.
	/// Use camera rotations in the x and y planes as observations.
	/// </summary>
	/// <param name="sensor">Vector sensor to add observations to.</param>
	public override void CollectObservations(VectorSensor sensor)
	{
		// If the agent showld use the camera rotations as observations.
		if (useVectorObservations)
		{
			// Add left eye rotation as observations.
			sensor.AddObservation(leftEyeCamera.transform.localRotation.eulerAngles.x);
			sensor.AddObservation(leftEyeCamera.transform.localRotation.eulerAngles.z);

			// Add right eye rotation as observations.
			sensor.AddObservation(leftEyeCamera.transform.localRotation.eulerAngles.x);
			sensor.AddObservation(leftEyeCamera.transform.localRotation.eulerAngles.z);
		}
	}

	/// <summary>
	/// 
	/// </summary>
	/// <param name="actionMask"></param>
	public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
	{
		// TODO Mask actions that would cause the agent to pass the camera rotation limitations.
	}

	/// <summary>
	/// Receibe actions and process them for eye control.
	/// </summary>
	/// <param name="actionBuffers">Actions buffer to get actions from.</param>
	public override void OnActionReceived(ActionBuffers actionBuffers)
	{
		// Get left and right eye action buffers for the X and Y directions.
		var leftEyeActionX = actionBuffers.DiscreteActions[0];
		var leftEyeActionZ = actionBuffers.DiscreteActions[1];
		var rightEyeActionX = actionBuffers.DiscreteActions[2];
		var rightEyeActionZ = actionBuffers.DiscreteActions[3];
		
		// Rotate each camera by a given direction with the corresponding rotation action.
		RotateCamera(leftEyeCamera, Vector3.right, (CameraRotationActions) leftEyeActionX);
		RotateCamera(leftEyeCamera, Vector3.up, (CameraRotationActions) leftEyeActionZ);
		RotateCamera(rightEyeCamera, Vector3.right, (CameraRotationActions) rightEyeActionX);
		RotateCamera(rightEyeCamera, Vector3.up, (CameraRotationActions) rightEyeActionZ);

		// Check if the target is in the frustrum for both the left and right cameras.
		bool leftEyeOnTarget = TargetInCameraFrustrum(m_TargetMeshRenderer, leftEyeCamera);
		bool rightEyeOnTarget = TargetInCameraFrustrum(m_TargetMeshRenderer, rightEyeCamera);

		// Reward the agent if the target comes into view of either eye.
		// Unreward the agent if the target comes out of view of either eye.
		AddReward(System.Convert.ToInt32(leftEyeOnTarget) - System.Convert.ToInt32(m_LeftEyeWasOnTarget));
		AddReward(System.Convert.ToInt32(rightEyeOnTarget) - System.Convert.ToInt32(m_RightEyeWasOnTarget));

        // If both eyes are on target.
        if (leftEyeOnTarget && rightEyeOnTarget)
        {
			// Add to the counter the amount of time that the agent has kept the target on view.
			m_TimeOnTarget = Time.deltaTime;
        }

		// If the agent doesn't ahve both eyes on the target.
		else
		{
			// Remove reward from agent the longer it takes to find the target.
            AddReward(-0.01f);

			// Clear the eye sight timer.
            m_TimeOnTarget = 0.0f;
		}

        // If the agent keeps the target on sight with both eyes for at least eyeOnTargetTime seconds.
		if (m_TimeOnTarget >= eyeOnTargetTime)
		{
            // Reward the agent for completing the episode.
            AddReward(2.0f);

            // End the episode.
            EndEpisode();
		}

        // Set the previous state for if the agent has either eye on the target.
        m_LeftEyeWasOnTarget = leftEyeOnTarget;
		m_RightEyeWasOnTarget = rightEyeOnTarget;
    }

	/// <summary>
	/// For every episode, set the agent and the target.
	/// </summary>
	public override void OnEpisodeBegin()
	{
        // Set the previous state for if the agent has either eye on the target to false.
        m_LeftEyeWasOnTarget = false;
        m_RightEyeWasOnTarget = false;

        // Clear the eye sight timer.
        m_TimeOnTarget = 0.0f;

        // Set the agent.
        SetAgent();

		// Set the target.
		SetTarget();
	}

	/// <summary>
	/// Heuristic method of controlling agent motion.
	/// Set aggent eye motion from the arrow keys (X = horizontal arrows, Y + vertical arrows).
	/// </summary>
	/// <param name="actionsOut"></param>
	public override void Heuristic(in ActionBuffers actionsOut)
	{
		// Get descrete actions from the ActionBuffers.
		ActionSegment<int> descreteActions = actionsOut.DiscreteActions;

        // Set discrete actions for the X component of rotation to the Horizontal arrows.
        // Set discrete actions for the Y component of rotation to the Vertical arrows.
        descreteActions[0] = (int) Input.GetAxisRaw("Horizontal") + 1;
		descreteActions[1] = (int) Input.GetAxisRaw("Vertical") + 1;
		descreteActions[2] = (int) Input.GetAxisRaw("Horizontal") + 1;
		descreteActions[3] = (int) Input.GetAxisRaw("Vertical") + 1;
    }
	#endregion

	#region Other Methods
	/// <summary>
	/// Randomice agent eyes' orientation.
	/// </summary>
	private void SetAgent()
	{
		// Rotate left camera to random angles in the x and y planes.
		leftEyeCamera.transform.Rotate(Vector3.right, Random.Range(-180f, 180f));
		leftEyeCamera.transform.Rotate(Vector3.up, Random.Range(-180f, 180f));

		// Rotate right camera to random angles in the x and y planes.
		rightEyeCamera.transform.Rotate(Vector3.right, Random.Range(-180f, 180f));
		rightEyeCamera.transform.Rotate(Vector3.up, Random.Range(-180f, 180f));
	}

	/// <summary>
	/// Randomice target position on backgrownd, change it's color, and change it's shape.
	/// </summary>
	public void SetTarget()
	{
		// TODO Randomize target position on backgrownd
		// TODO Change target color
		// TODO Change target mesh        
	}

	/// <summary>
	/// Rotate a given camera with a direction vector and the corresponding action.
	/// </summary>
	/// <param name="camera">The camera to rotate.</param>
	/// <param name="direction">The direction Vector3.</param>
	/// <param name="action">The corresponding action (positive rotation, negative rotation, or do nothing).</param>
	void RotateCamera(Camera camera, Vector3 direction, CameraRotationActions action)
    {
		// On a given camera control action action select.
		switch (action)
		{
			case CameraRotationActions.Nothing:
				// do nothing
				break;

			case CameraRotationActions.Positive:
                camera.transform.Rotate(direction, eyeRotationSpeed * Time.deltaTime);
				break;

			case CameraRotationActions.Negative:
                camera.transform.Rotate(-direction, eyeRotationSpeed * Time.deltaTime);
				break;
		}
	}

    /// <summary>
    /// Check if the target's AABB from the target's bounds are utside the camera's frustrum. 
    /// </summary>
    /// <param name="renderer">The target's mesh renderer.</param>
    /// <param name="camera">The camere to check for if the target is within its frustrum.</param>
    /// <returns>True if the object is fully within the camera's frustrum, false if otherwise.</returns>
    bool TargetInCameraFrustrum(Renderer renderer, Camera camera)
	{
        // Get the frustrum planes for the left and right eye cameras.
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(leftEyeCamera);

		// Iterate over each plane.
        for (int i = 0; i < planes.Length; i++)
        {
			// Flip each plane.
            planes[i] = planes[i].flipped;
        }

		// Return if the AABB planes from the target's bounds are outside the camera's frustrum planes.
		return !GeometryUtility.TestPlanesAABB(planes, renderer.bounds); ;
    }

	/// <summary>
	/// Update method called once per frame.
	/// </summary>
	public void FixedUpdate()
	{
		// If both cameras are set and there is a graphical device to render
		if (leftEyeCamera != null 
			&& rightEyeCamera != null 
			&& SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null)
		{
			// Force render both cameras.
			leftEyeCamera.Render();
			rightEyeCamera.Render();
		}

		// If not on inference mode.
		if (Academy.Instance.IsCommunicatorOn)
		{
			RequestDecision();
		}

		// If on inference mode.
		else
		{
			// If the time required between desicions has passed.
			if (m_TimeSinceDecision >= timeBetweenDecisionsAtInference)
			{
				// Manually request a desicion.
				RequestDecision();

				// Reset time between decisions.
				m_TimeSinceDecision = 0f;
			}

			// If the time between desicions hasn't passed.
			else
			{
				// Add the time between fixed updates to the time between desicions.
				m_TimeSinceDecision += Time.fixedDeltaTime;
			}
		}
	}
	#endregion
}
