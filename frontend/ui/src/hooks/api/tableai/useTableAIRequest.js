import { useEffect, useState } from 'react';

export function useTableAIRequest(endpoint, options = {}) {
  const { dependency = null, params = {}, transform = (d) => d } = options;

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const url = typeof endpoint === 'function' ? endpoint(dependency) : endpoint;

    if (!url) return;

    setLoading(true);
    setError(null);

    fetch(url, params)
      .then((res) => {
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        return res.json();
      })
      .then((json) => {
        setData(transform(json));
        setLoading(false);
      })
      .catch((err) => {
        console.error('TableAI request error:', err);
        setError(err);
        setLoading(false);
      });
  }, [endpoint, dependency]);

  return { data, loading, error };
}