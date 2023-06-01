using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using Random = UnityEngine.Random;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;

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

	[Tooltip("Angles used to limit left eye camera rotation.")]
	public Vector2 leftEyeRestrictionAngle;

	[Tooltip("Angles used to limit right eye camera rotation.")]
	public Vector2 rightEyeRestrictionAngle;

	[Tooltip("Controls the speed at wich the agent is able to rotate each eye, in degrees per second.")]
	[Range(10.0f, 180.0f)]
	public float eyeRotationSpeed;

    [Tooltip("Controls the amount of time the target agent showld keep eyesight on the target before goal.")]
    [Range(0.0f, 10.0f)]
    public float eyeOnTargetTime;

	[Header("On inference")]
	[Tooltip("Time between desicions on inference mode.")]
	[Range(0.05f, 0.5f)]
	public float timeBetweenDecisionsAtInference;
	#endregion

	#region Member Parameters
	/// <summary>
	/// Reward coeficient calculated before the start of training.
	/// Adjust rewards to ensure on goal completion agent's reward at least breck even.
	/// </summary>
	float m_RewardCoeficient;

	/// <summary>
	/// Time since last desicion was taken in inference mode.
	/// </summary>
	float m_TimeSinceDecision;

	/// <summary>
	/// The amount of time the agent has stayed on target.
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
    /// Wheter the left camera is partially focused on the target or not.
    /// </summary>
    bool m_LeftEyeWasPartiallyOnTarget;

    /// <summary>
    /// Wheter the right camera is partially focused on the target or not.
    /// </summary>
    bool m_RightEyeWasPartiallyOnTarget;

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
        // If on inference mode or heuristic.
        if (!Academy.Instance.IsCommunicatorOn)
        {
			// Set the max step to zero.
            this.MaxStep = 0;
        }

		// Calculate reward coeficient based on max step, equivalent to the amount of seconds for full episode.
		m_RewardCoeficient = MaxStep / 50;

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
	/// Also use camera restriction angles.
	/// </summary>
	/// <param name="sensor">Vector sensor to add observations to.</param>
	public override void CollectObservations(VectorSensor sensor)
	{
		// Add left eye rotation as observations.
		sensor.AddObservation(WrapAngle(leftEyeCamera.transform.localRotation.eulerAngles.x));
		sensor.AddObservation(WrapAngle(leftEyeCamera.transform.localRotation.eulerAngles.y));

		// Add right eye rotation as observations.
		sensor.AddObservation(WrapAngle(leftEyeCamera.transform.localRotation.eulerAngles.x));
		sensor.AddObservation(WrapAngle(leftEyeCamera.transform.localRotation.eulerAngles.y));

        // Add left eye restriction angles as observations.
        sensor.AddObservation(leftEyeRestrictionAngle.x);
        sensor.AddObservation(leftEyeRestrictionAngle.y);

        // Add right eye restriction angles as observations.
        sensor.AddObservation(rightEyeRestrictionAngle.x);
        sensor.AddObservation(rightEyeRestrictionAngle.y);
	}

	/// <summary>
	/// Receibe actions and process them for eye control.
	/// </summary>
	/// <param name="actionBuffers">Actions buffer to get actions from.</param>
	public override void OnActionReceived(ActionBuffers actionBuffers)
	{
		// Get left and right eye action buffers for the X and Y directions.
		var leftEyeActionX = actionBuffers.DiscreteActions[0];
		var leftEyeActionY = actionBuffers.DiscreteActions[1];
		var rightEyeActionX = actionBuffers.DiscreteActions[2];
		var rightEyeActionY = actionBuffers.DiscreteActions[3];
		
		// Rotate each camera by a given direction with the corresponding rotation action.
		RotateCamera(leftEyeCamera, Vector3.right, (CameraRotationActions) leftEyeActionX);
		RotateCamera(leftEyeCamera, Vector3.up, (CameraRotationActions) leftEyeActionY);
		RotateCamera(rightEyeCamera, Vector3.right, (CameraRotationActions) rightEyeActionX);
		RotateCamera(rightEyeCamera, Vector3.up, (CameraRotationActions) rightEyeActionY);

        // Clamp camera rotations to the corresponding restriction angles.
        ClampCameraRotation(leftEyeCamera, leftEyeRestrictionAngle.x, leftEyeRestrictionAngle.y);
        ClampCameraRotation(rightEyeCamera, rightEyeRestrictionAngle.x, rightEyeRestrictionAngle.y);

        // Check if the target is in the frustum for both the left and right cameras.
        bool leftEyeOnTarget = TargetInCameraFrustum(m_TargetMeshRenderer, leftEyeCamera);
		bool rightEyeOnTarget = TargetInCameraFrustum(m_TargetMeshRenderer, rightEyeCamera);

        // Check if the target is partially in the frustum for both the left and right cameras.
        bool leftEyePartiallyOnTarget = TargetPartiallyInCameraFrustrum(m_TargetMeshRenderer, leftEyeCamera);
        bool rightEyePartiallyOnTarget = TargetPartiallyInCameraFrustrum(m_TargetMeshRenderer, rightEyeCamera);

        // Reward the agent if the target comes into view of either eye.
        // Unreward the agent if the target comes out of view of either eye.
        AddReward((System.Convert.ToInt32(leftEyeOnTarget) - System.Convert.ToInt32(m_LeftEyeWasOnTarget)) * 0.1f * m_RewardCoeficient);
		AddReward((System.Convert.ToInt32(rightEyeOnTarget) - System.Convert.ToInt32(m_RightEyeWasOnTarget)) * 0.1f * m_RewardCoeficient);

        // Reward the agent if the target comes partially into view of either eye.
        // Unreward the agent if the target comes out of view of either eye.
        AddReward((System.Convert.ToInt32(leftEyePartiallyOnTarget) - System.Convert.ToInt32(m_LeftEyeWasPartiallyOnTarget)) * 0.05f * m_RewardCoeficient);
        AddReward((System.Convert.ToInt32(rightEyePartiallyOnTarget) - System.Convert.ToInt32(m_RightEyeWasPartiallyOnTarget)) * 0.05f * m_RewardCoeficient);

        // If both eyes are on target.
        if (leftEyeOnTarget && rightEyeOnTarget)
        {
			// Add to the counter the amount of time that the agent has kept the target on view.
			m_TimeOnTarget += Time.deltaTime;
        }

		// If the agent doesn't have both eyes on the target.
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
            AddReward(0.2f * m_RewardCoeficient);

            // End the episode.
            EndEpisode(); 
        }

        // Set the previous state for if the agent has either eye on the target.
        m_LeftEyeWasOnTarget = leftEyeOnTarget;
		m_RightEyeWasOnTarget = rightEyeOnTarget;
		m_LeftEyeWasPartiallyOnTarget = leftEyePartiallyOnTarget;
		m_RightEyeWasPartiallyOnTarget = rightEyePartiallyOnTarget;
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
	void SetAgent()
	{
		// Rotate left camera to random angles in the x and y planes.
		leftEyeCamera.transform.Rotate(Vector3.right, Random.Range(-leftEyeRestrictionAngle.x, leftEyeRestrictionAngle.x));
		leftEyeCamera.transform.Rotate(Vector3.up, Random.Range(-leftEyeRestrictionAngle.y, leftEyeRestrictionAngle.y));

		// Rotate right camera to random angles in the x and y planes.
		rightEyeCamera.transform.Rotate(Vector3.right, Random.Range(-rightEyeRestrictionAngle.x, rightEyeRestrictionAngle.x));
		rightEyeCamera.transform.Rotate(Vector3.up, Random.Range(-rightEyeRestrictionAngle.y, rightEyeRestrictionAngle.y));
	}

	/// <summary>
	/// Randomice target position on backgrownd.
	/// </summary>
	void SetTarget()
	{
		// Randomice position in y and z.
		target.transform.localPosition = new Vector3(5, Random.Range(-4, 4), Random.Range(-4, 4));
	}

	/// <summary>
	/// Rotate a given camera with a direction vector and the corresponding action.
	/// </summary>
	/// <param name="camera">The camera to rotate.</param>
	/// <param name="direction">The direction Vector3.</param>
	/// <param name="action">The corresponding action (positive rotation, negative rotation, or do nothing).</param>
	void RotateCamera(Camera camera, Vector3 direction, CameraRotationActions action)
    {
		// On a given camera apply the control action.
		switch (action)
		{
			case CameraRotationActions.Nothing:
				// do nothing
				break;

			case CameraRotationActions.Positive:
				// Rotate camera in the given direction.
                camera.transform.Rotate(direction, eyeRotationSpeed * Time.deltaTime);
				break;

			case CameraRotationActions.Negative:
				// Rotate camera in the oposite the given direction.
                camera.transform.Rotate(-direction, eyeRotationSpeed * Time.deltaTime);
				break;
		}
	}

    /// <summary>
    /// Clamp the camera's x and y angles, by the given angles (restricted to: -angle, angle).
	/// The z angle is left as zero.
    /// </summary>
    /// <param name="camera">The camera to rotate.</param>
    /// <param name="angleX">The angle in X to clamp to.</param>
    /// <param name="angleY">The angle in Y to clamp to.</param>
    void ClampCameraRotation(Camera camera, float angleX, float angleY)
    {
        // Get the clamped camara's rotation angles in the x, y, and z directions.
        float rotationX = Mathf.Clamp(WrapAngle(camera.transform.localRotation.eulerAngles.x), -angleX, angleX);
        float rotationY = Mathf.Clamp(WrapAngle(camera.transform.localRotation.eulerAngles.y), -angleY, angleY);

        // Fix the camera's rotation to the clamped angles (leave z angle as zero).
        camera.transform.localRotation = Quaternion.Euler(UnwrapAngle(rotationX), UnwrapAngle(rotationY), 0);
    }

    /// <summary>
    /// Pass Euler angles to a range of (-180, 180].
    /// </summary>
    /// <param name="angle">The angle to wrap.</param>
    /// <returns>The wraped angle in range (-180, 180].</returns>
    static float WrapAngle(float angle)
    {
		// Get the angle modulo 360.
        angle %= 360;

		// If the angle is greater than 180 degrees.
        if (angle > 180)
			// Return the angle as if starting from the other side of the cincunference.
            return angle - 360;

		// Otherwise, return the angle as is.
        return angle;
    }

    /// <summary>
    /// Pass angles in range (-180, 180] to Euler angles (range [0, 360)).
    /// </summary>
    /// <param name="angle">The angle to wrap.</param>
    /// <returns>The wraped angle in range [0, 360).</returns>
    static float UnwrapAngle(float angle)
    {
        // Get the angle modulo 360.
        angle %= 360;

        // If the angle is greater than zero.
        if (angle >= 0)
            // Return the angle as is.
            return angle;

        // Otherwise, return the angle as if starting from the other side of the cincunference. 
        return 360 + angle;
    }

    /// <summary>
    /// Check if the target's AABB from the target's bounds are outside the camera's frustum. 
    /// </summary>
    /// <param name="renderer">The target's mesh renderer.</param>
    /// <param name="camera">The camere to check for if the target is within its frustum.</param>
    /// <returns>True if the object is fully within the camera's frustum, false if otherwise.</returns>
    bool TargetInCameraFrustum(Renderer renderer, Camera camera)
	{
        // Get the frustum planes for the left and right eye cameras.
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(leftEyeCamera);

		// Initialice condition on true. 
		bool isInFrustrum = true;

		// Iterate over each plane.
		for (int i = 0; i < planes.Length; i++)
		{
            // Flip the plane.
            Plane plane = planes[i].flipped;

            // Get the point on the edge of the bounding "elipsoid":
            // closest to the plane when the center is on the oposite side of the plane's normal,
            // and furthest to the plane when the center is on the same side of the plane's normal.
            Vector3 closestPoint = renderer.bounds.center + (plane.normal.normalized * renderer.bounds.extents.magnitude);

            // If the closest point is not on the same side as the plane's normal, "and" the result.
            isInFrustrum &= !plane.GetSide(closestPoint);
        }
		
		// Return if the target's bounds are in the camera's frustum planes.
		return isInFrustrum;
    }

    /// <summary>
    /// Check if the target's AABB from the target's bounds are outside the camera's frustum. 
    /// </summary>
    /// <param name="renderer">The target's mesh renderer.</param>
    /// <param name="camera">The camere to check for if the target is within its frustum.</param>
    /// <returns>True if the object is partially within the camera's frustum, false if otherwise.</returns>
    bool TargetPartiallyInCameraFrustrum(Renderer renderer, Camera camera)
    {
        // Get the frustum planes for the left and right eye cameras.
        Plane[] planes = GeometryUtility.CalculateFrustumPlanes(camera);
		
        // Return if the AABB planes from the target's bounds are inside the camera's frustum planes.
        return GeometryUtility.TestPlanesAABB(planes, renderer.bounds);
    }

	/// <summary>
	/// Update method called once per frame.
	/// </summary>
	public void FixedUpdate()
	{
		// If both cameras are set and there is a graphical device to render.
		if (leftEyeCamera != null 
			&& rightEyeCamera != null 
			&& SystemInfo.graphicsDeviceType != GraphicsDeviceType.Null)
		{
			// Force render both cameras.
			leftEyeCamera.Render();
			rightEyeCamera.Render();
		}

		// If not on inference mode or heuristic.
		if (Academy.Instance.IsCommunicatorOn)
		{
            // Manually request a desicion.
            RequestDecision();
		}

		// If on inference mode or heuristic.
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
